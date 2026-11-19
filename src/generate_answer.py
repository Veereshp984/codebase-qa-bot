
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
        temperature=0
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
{chat_history}

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
"""

    human_prompt = f"""
Retrieved Code Context:

{context}

User Question:
{query}
"""

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL"),
        temperature=0
    )

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ])

    return response.content


# 3. Main execution


def main():

    print("Starting Codebase Q&A Bot...")

    vector_store = load_vector_store()

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
    main()

