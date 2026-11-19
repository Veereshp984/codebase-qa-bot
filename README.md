# Codebase Q&A Bot 🤖

An AI-powered chatbot that helps developers understand and explore GitHub repositories by asking questions in natural language. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant code snippets and generate answers grounded in the repository's source code.

Instead of manually searching through multiple files, developers can connect a GitHub repository and ask questions about its implementation, functions, authentication flow, APIs, and other code-related details.

## ✨ Features

- **GitHub Repository Integration:** Clone a GitHub repository directly using its URL.
- **Automated Code Ingestion:** Load supported source files and extract their contents and metadata.
- **Language-Aware Chunking:** Split source code into manageable chunks while preserving useful code structure.
- **Vector Embeddings:** Generate semantic embeddings using the Sentence Transformers model.
- **Vector Database:** Store code chunks, embeddings, and metadata in ChromaDB.
- **Semantic Search:** Retrieve relevant code snippets based on natural-language questions.
- **MMR Retrieval Support:** Support Maximal Marginal Relevance (MMR) retrieval for balancing relevance and diversity.
- **Context Expansion:** Retrieve neighbouring chunks to provide additional context around matching code.
- **AI-Powered Answers:** Use Groq-hosted LLMs to generate answers based on retrieved code.
- **Conversation Context:** Rewrite follow-up questions using recent chat history.
- **Source References:** Display the source files used to generate an answer.
- **Repository Explorer:** Browse repository files and view their contents with syntax highlighting.
- **Interactive UI:** Use a Streamlit interface with a chat panel and repository explorer.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LangChain | RAG pipeline and component integration |
| Groq | LLM inference and answer generation |
| HuggingFace Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector storage and similarity search |
| Streamlit | Interactive user interface |
| Git | Repository cloning and version control |

## 🏗️ Architecture

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

```text
                  GitHub Repository
                          |
                          v
                    Clone Repository
                          |
                          v
                    Load Source Files
                          |
                          v
                    Split into Chunks
                          |
                          v
                 Generate Embeddings
                          |
                          v
                    ChromaDB Store
                          |
                          |
       User Question    |
             |          |
             v          |
       Rewrite Question |
             |          |
             v          |
       Semantic Retrieval
             |
             v
       Retrieve Relevant Chunks
             |
             v
       Expand Neighbouring Chunks
             |
             v
       Build Context
             |
             v
       Groq LLM
             |
             v
       Generate Answer
             |
             v
       Display Answer and Sources
```

## 📂 Project Structure

```text
codebase-qa-bot/
│
├── app.py
│
├── src/
│   ├── clone_repo.py
│   ├── load_files.py
│   ├── split_code.py
│   ├── experiment_chunking.py
│   ├── build_store.py
│   ├── retrieve.py
│   └── generate_answer.py
│
├── repos/                 # Cloned repositories
│
├── db/                    # ChromaDB vector databases
│
├── eval/                  # Retrieval evaluation
│
├── .env                   # API credentials (not committed)
├── .env.example           # Environment variable template
├── .gitignore
├── requirements.txt
├── NOTES.md
└── README.md
```

## ⚙️ How It Works

### 1. Repository Cloning

The user provides a GitHub repository URL. The application uses Python's `subprocess` module to execute Git and clone the repository locally.

A shallow clone (`--depth 1`) is used to avoid downloading the complete Git commit history.

### 2. File Loading

The application scans the cloned repository and loads supported file types:

- Python (`.py`)
- JavaScript (`.js`)
- TypeScript (`.ts`)
- Java (`.java`)
- Markdown (`.md`)

It skips unnecessary directories and files such as `.git`, `node_modules`, virtual environments, and lock files.

Each loaded file is converted into a LangChain Document containing its content and metadata, including its source path and programming language.

### 3. Code Chunking

Large source files are divided into smaller chunks so that they can be processed efficiently.

The application uses a language-aware recursive text splitter with the following configuration:

- Chunk size: 1000 characters
- Chunk overlap: 100 characters

Chunk metadata includes the source file path, programming language, and chunk index.

### 4. Embedding Generation

Each chunk is converted into a numerical vector using the HuggingFace model:

`sentence-transformers/all-MiniLM-L6-v2`

The model generates 384-dimensional embeddings, which represent the semantic meaning of the code and text.

### 5. Vector Storage

The generated embeddings, original text chunks, and metadata are stored in ChromaDB.

Each repository has its own local vector database, allowing the application to load an existing index instead of rebuilding it every time.

### 6. Retrieval

When a user asks a question:

1. The question is optionally rewritten using recent conversation history.
2. The query is converted into an embedding.
3. ChromaDB performs similarity search to retrieve relevant code chunks.
4. Neighbouring chunks are retrieved to provide additional context.

### 7. Answer Generation

The retrieved code snippets are passed to the Groq LLM along with a grounding prompt.

The prompt instructs the model to:
- Answer using the retrieved code.
- Avoid inventing implementation details.
- Mention relevant source file paths.
- Clearly state when the retrieved information is insufficient.

The generated answer is then displayed in the Streamlit interface.

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.10 or later
- Git
- A Groq API key

### 1. Clone This Project

```bash
git clone https://github.com/<your-username>/<your-repository>.git

cd codebase-qa-bot
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model_name
```

Replace the placeholder values with your Groq API key and the model name you want to use.

You can obtain an API key from [Groq Console](https://console.groq.com/).

**Important:** Never commit your `.env` file or expose your API key publicly.

### 5. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💬 How to Use

1. Start the application.
2. Open the sidebar.
3. Enter a GitHub repository URL.
4. Click **Clone and Index**.
5. Wait for the repository to be cloned and indexed.
6. Ask questions about the repository in the chat panel.
7. Review the generated answers and source file references.
8. Use the Repository Explorer to browse source files and inspect their contents.

### Example Questions

You can ask questions such as:

- How does user authentication work in this project?
- Where is the JWT token generated?
- Explain the login flow step by step.
- How is the database connection configured?
- Which function handles user registration?
- How does the application handle invalid credentials?

## 🔐 Environment and Data

- Repository cloning and vector database storage happen locally.
- The `.env` file stores API credentials and should remain private.
- The embedding model runs locally through Sentence Transformers.
- Groq is used for LLM inference, so retrieved context is sent to the configured Groq service for answer generation.

## 🔮 Future Improvements

Possible improvements include:

- Improve retrieval accuracy through systematic evaluation.
- Add support for more programming languages.
- Improve source-level citations with line references.
- Add more advanced retrieval strategies.
- Improve handling of large repositories.
- Add automated testing for the complete RAG pipeline.

## 🎯 Project Objective

The main objective of this project is to understand and implement a practical RAG application for software development.

It brings together repository ingestion, code chunking, embedding generation, vector databases, semantic retrieval, context management, and LLM-based answer generation to make codebase exploration easier.

---

**Built with Python, LangChain, ChromaDB, HuggingFace, Groq and Streamlit.**
