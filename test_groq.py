import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    temperature=0
)

response = llm.invoke(
    "Explain what RAG is in simple words."
)

print(response.content)