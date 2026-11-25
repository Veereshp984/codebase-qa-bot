from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

sentence = "Python is a programming language"

vector = embedding_model.embed_query(sentence)

print("Embedding:", vector)
print("Dimensions:", len(vector))
print("First five values:", vector[:5])