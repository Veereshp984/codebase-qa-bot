
import os
import re

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieve import (
    load_vector_store,
    retrieve_function_by_name,
    find_function_source
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

    query = "How does get_my_bookings retrieve booking records using user_id?"

    print("\nUser Question:", query)

    # Step 1: Extract function name
    match = re.search(
        r"\b[a-zA-Z_]\w*_[a-zA-Z_]\w*\b",
        query
    )

    function_name = match.group(0) if match else None

    if not function_name:
        print("\nNo function name found in the question.")
        return

    print("\nDetected function:", function_name)

    # Step 2: Find source file automatically
    source = find_function_source(
        vector_store=vector_store,
        function_name=function_name
    )

    if not source:
        print("\nFunction not found in the repository.")
        return

    print("\nDiscovered source:", source)

    # Step 3: Retrieve complete function
    function_chunks = retrieve_function_by_name(
        vector_store=vector_store,
        function_name=function_name,
        source=source
    )

    if not function_chunks:
        print("\nCould not retrieve function implementation.")
        return

    print(
        "\nFunction chunks retrieved:",
        len(function_chunks)
    )

    for doc in function_chunks:
        print("Chunk:", doc.metadata["chunk_index"])

    # Step 4: Generate answer
    answer = generate_answer(
        query,
        [(doc, None) for doc in function_chunks]
    )

    print("\n==============================")
    print("BOT ANSWER")
    print("==============================")
    print(answer)


if __name__ == "__main__":
    main()
