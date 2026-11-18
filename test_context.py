import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL"),
    temperature=0
)

# Three code chunks
chunks_3 = [
    "Chunk 1: The application uses FastAPI.",
    "Chunk 2: Authentication is handled using JWT.",
    "Chunk 3: User passwords are hashed before storage."
]

# Ten code chunks
chunks_10 = chunks_3 + [
    "Chunk 4: Database connection uses PostgreSQL.",
    "Chunk 5: SQLAlchemy is used as ORM.",
    "Chunk 6: Users can register using email.",
    "Chunk 7: Access tokens are generated after login.",
    "Chunk 8: Password hashing uses bcrypt.",
    "Chunk 9: Environment variables are stored in .env.",
    "Chunk 10: Alembic manages database migrations."
]

question = "Explain how authentication works in this application."

def ask_llm(chunks):
    context = "\n".join(chunks)

    prompt = f"""
    Answer the question using only the given context.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)
    print(response.content)


print("========== WITH 3 CHUNKS ==========")
ask_llm(chunks_3)

print("\n========== WITH 10 CHUNKS ==========")
ask_llm(chunks_10)