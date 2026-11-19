
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieve import (
    load_vector_store,
    retrieve_chunks,
    get_neighbouring_chunks
)

load_dotenv()


# 1. Format retrieved code chunks
def format_context(retrieved_chunks):

    context = []

    for doc, score in retrieved_chunks:

        source = doc.metadata["source"]

        chunk_text = f"""
File: {source}

Code:
{doc.page_content}
"""

        context.append(chunk_text)

    return "\n\n".join(context)


def rewrite_question(query, chat_history):

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL"),
        temperature=0,
        max_tokens=128
    )

    prompt = f"""
You are a question rewriting assistant.

Your task is to rewrite the latest user question
so that it can be understood independently.

Use the conversation history to understand
what the user is referring to.

Do not answer the question.
Only return the rewritten question.

Conversation history:
{chat_history[-4:]}

Latest question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content.strip()


# 2. Generate answer using Groq
def generate_answer(query, retrieved_chunks):

    context = format_context(retrieved_chunks)

    MAX_CONTEXT_CHARS = 20000
    context = context[:MAX_CONTEXT_CHARS]

    system_prompt = """
You are a codebase question-answering assistant.
Answer the user's question using ONLY the provided code snippets.

Rules:
1. Do not use outside knowledge.
2. Do not make up information.
3. Do not infer function behavior from its name.
4. If the provided snippets do not contain enough information,
   respond with:
   I don't have enough information in the retrieved code.
5. Explain the code clearly.
6. Mention the relevant file paths in your answer.
7. Distinguish actual implementation from comments or descriptions.
8. Keep answers concise and focused on the user's question.
9. Prefer bullet points over long paragraphs.
10. Do not repeat the same information.
11. Include code snippets only when they help explain the answer.
12. For simple questions, answer in 3-5 sentences.

13. Never present README descriptions or test cases as actual implementation.
14. Clearly distinguish implementation code, tests, and documentation.
15. Do not claim a function performs an operation unless the retrieved
    code shows it.
16. If documentation describes a feature but its implementation
    is missing, explicitly say that the implementation was not retrieved.
17. Never invent missing code or fill gaps using assumptions.
18. Explain operations in the order they actually execute in the code.
19. Do not combine frontend and backend behavior unless the retrieved
    snippets establish the connection.
20. For every important claim, identify the code that supports it.
21. If a step is missing from the retrieved context, explicitly identify
    that missing step instead of assuming what happens.
22. Never describe inferred behavior as confirmed implementation.
"""

    human_prompt = f"""
Retrieved Code Context:

{context}

User Question:
{query}
"""

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL"),
        temperature=0,
        max_tokens=512
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return response.content


# 3. Main execution


def main(repo_name):

    print("Starting Codebase Q&A Bot...")

    vector_store = load_vector_store(repo_name)

    # Store the conversation
    chat_history = []

    while True:

        query = input("\nYou: ")

        # Exit condition
        if query.lower() == "quit":
            print("Chat ended!")
            break

        # Rewrite question using previous conversation
        rewritten_query = rewrite_question(
            query=query,
            chat_history=chat_history
        )

        print("\nRewritten Question:", rewritten_query)

        # Retrieve relevant chunks
        retrieved_chunks = retrieve_chunks(
            vector_store=vector_store,
            query=rewritten_query,
            k=6,
            language="python",
            use_mmr=False
        )
        
        print("\nRetrieved chunks for follow-up:")

        for doc, score in retrieved_chunks:
            print("\nSource:", doc.metadata["source"])
            print("Chunk:", doc.metadata["chunk_index"])
            print("Content:", doc.page_content[:300])

        # Expand neighbouring chunks
        expanded_chunks = get_neighbouring_chunks(
            vector_store=vector_store,
            retrieved_chunks=retrieved_chunks,
            window=1
        )

        # Generate answer using original question
        answer = generate_answer(
            query,
            [(doc, None) for doc in expanded_chunks]
        )

        print("\nBot:", answer)

        # Display unique source files
        source_paths = sorted({
            doc.metadata["source"]
            for doc in expanded_chunks
        })

        print("\nSources:")
        for source in source_paths:
            print("-", source)

        # Save conversation
        chat_history.append({
            "role": "user",
            "content": query
        })

        chat_history.append({
            "role": "assistant",
            "content": answer
        })


if __name__ == "__main__":

    repo_name = input("Enter repository name: ")

    main(repo_name)
