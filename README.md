# Codebase Q&A Bot 🤖

An AI-powered chatbot that helps developers understand and explore GitHub repositories using natural language. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant code snippets and generate answers grounded in the repository's source code.

Instead of manually searching through multiple files, developers can connect a GitHub repository and ask questions about its implementation, functions, authentication flows, APIs, and other code-related details.

## ✨ Features

- **GitHub Repository Integration:** Clone GitHub repositories directly using their URLs.
- **Automated Code Ingestion:** Load supported source files and extract their contents and metadata.
- **Language-Aware Chunking:** Split source code into manageable chunks while preserving useful code structure.
- **Vector Embeddings:** Generate semantic embeddings using HuggingFace Sentence Transformers.
- **PostgreSQL + pgvector:** Store code chunks, embeddings, and metadata in PostgreSQL.
- **HNSW Indexing:** Use approximate nearest-neighbour indexing for vector similarity search.
- **Semantic Search:** Retrieve relevant code snippets using cosine distance.
- **Repository Isolation:** Store and retrieve chunks using repository-specific identifiers.
- **Context Expansion:** Retrieve neighbouring chunks to provide additional context around matching code.
- **AI-Powered Answers:** Use Groq-hosted LLMs to generate answers based on retrieved code.
- **Conversation Context:** Rewrite follow-up questions using recent chat history.
- **Source References:** Display source files used to generate answers.
- **Retrieval Evaluation:** Evaluate retrieval performance using a 20-question dataset and Hit Rate@3.
- **Interactive UI:** Use a Streamlit interface with a chat panel and repository explorer.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application development |
| LangChain | RAG pipeline and component integration |
| Groq | LLM inference and answer generation |
| HuggingFace Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | 384-dimensional embedding model |
| PostgreSQL | Persistent relational and vector data storage |
| pgvector | Vector storage and similarity search |
| HNSW | Approximate nearest-neighbour indexing |
| Psycopg | PostgreSQL connectivity |
| Streamlit | Interactive user interface |
| Git | Repository cloning and version control |

## 🏗️ Architecture

The application follows a Retrieval-Augmented Generation (RAG) pipeline.

### Repository indexing

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
              Language-Aware
                Chunking
                     |
                     v
              Generate Embeddings
                     |
                     v
              PostgreSQL + pgvector
                     |
                     v
                 HNSW Index
```

### Question answering

```text
               User Question
                     |
                     v
             Rewrite Question
             (Chat History)
                     |
                     v
             Generate Query
                 Embedding
                     |
                     v
              Cosine Similarity
                  Search
                     |
                     v
             Retrieve Top-K Chunks
                     |
                     v
             Expand Neighbouring
                   Chunks
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
             Display Answer and
                 Sources
```

## 📂 Project Structure

```text
codebase-qa-bot/
│
├── app.py
├── chat.py
│
├── src/
│   ├── clone_repo.py
│   ├── load_files.py
│   ├── split_code.py
│   ├── experiment_chunking.py
│   ├── database.py
│   ├── postgres_store.py
│   └── generate_answer.py
│
├── tests/
│   ├── test_embeddings.py
│   ├── test_postgres_retrieval.py
│   ├── test_postgres_neighbours.py
│   └── test_query_plan.py
│
├── eval/
│   ├── evaluation_dataset.json
│   └── evaluate_retrieval.py
│
├── repos/                  # Locally cloned repositories
├── inspect_vectors.py
├── migrate_to_postgres.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── notes.md
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

Large source files are divided into smaller chunks so they can be processed efficiently.

The application uses a language-aware recursive text splitter with the following configuration:

- Chunk size: 1000 characters
- Chunk overlap: 100 characters

Each chunk contains metadata such as its source file path, programming language, and chunk index.

### 4. Embedding Generation

Each chunk is converted into a numerical vector using the HuggingFace model:

`sentence-transformers/all-MiniLM-L6-v2`

The model generates 384-dimensional embeddings representing the semantic meaning of the code and text.

The same embedding model is used to convert user questions into vectors during retrieval.

### 5. Vector Storage

The generated embeddings, original code chunks, and metadata are stored in PostgreSQL using pgvector.

The database uses a `VECTOR(384)` column to store embeddings.

Each chunk also contains a repository identifier (`repo_id`), allowing multiple repositories to be stored and queried independently.

An HNSW index using cosine distance is created to support approximate nearest-neighbour searches.

### 6. Retrieval

When a user asks a question:

1. The question is optionally rewritten using recent conversation history.
2. The rewritten question is converted into a 384-dimensional embedding.
3. PostgreSQL uses pgvector cosine distance (`<=>`) to retrieve the most relevant chunks.
4. Repository filtering ensures that retrieval is performed against the selected repository.
5. Neighbouring chunks are retrieved to provide additional context.

### 7. Answer Generation

The retrieved code snippets are passed to the Groq LLM along with a grounding prompt.

The prompt instructs the model to:

- Answer using only the retrieved code.
- Avoid inventing implementation details.
- Mention relevant source file paths.
- Distinguish implementation code from tests and documentation.
- Clearly state when the retrieved information is insufficient.

The generated answer and source references are displayed in the Streamlit interface.

## 🚀 Getting Started

### Prerequisites

Install the following:

- Python 3.10 or later
- Git
- PostgreSQL
- pgvector PostgreSQL extension
- A Groq API key

### 1. Clone This Project

```bash
git clone https://github.com/Veereshp984/codebase-qa-bot.git

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
python -m pip install -r requirements.txt
```

### 4. Configure PostgreSQL

Create a PostgreSQL database:

```sql
CREATE DATABASE codebase_qa;
```

Connect to the database and enable pgvector:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Create the code chunks table:

```sql
CREATE TABLE code_chunks (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding VECTOR(384) NOT NULL,
    repo_id TEXT
);
```

Create the HNSW index:

```sql
CREATE INDEX idx_code_chunks_embedding
ON code_chunks
USING hnsw (embedding vector_cosine_ops);
```

Make sure your PostgreSQL server is running and note your database connection details.

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=your_groq_model_name
DATABASE_URL=postgresql://username:password@localhost:5432/codebase_qa
```

Replace the placeholders with your actual credentials and PostgreSQL connection details.

You can obtain a Groq API key from [Groq Console](https://console.groq.com/).

**Important:** Never commit your `.env` file or expose your API key or database credentials publicly.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💬 How to Use

1. Start the Streamlit application.
2. Open the sidebar.
3. Enter a GitHub repository URL.
4. Click **Clone and Index**.
5. Wait for the repository to be cloned and indexed.
6. Ask questions about the repository in the chat panel.
7. Review generated answers and source file references.
8. Use the Repository Explorer to browse source files and inspect their contents.

### Example Questions

- How does user authentication work in this project?
- Where is the JWT token generated?
- Explain the login flow step by step.
- How is the database connection configured?
- Which function handles user registration?
- How does the application handle invalid credentials?

## 📊 Retrieval Evaluation

The retrieval pipeline was evaluated using a dataset of 20 questions with expected source files.

The evaluation metric used was **Hit Rate@3**.

Hit Rate@3 measures whether at least one expected source file appears among the top three retrieved results for a question.

| Metric | Result |
|---|---:|
| Evaluation questions | 20 |
| Successful hits | 20 |
| Misses | 0 |
| Hit Rate@3 | 100% |

The PostgreSQL + pgvector implementation achieved 20 successful hits out of 20 questions on the evaluation dataset.

This result measures source retrieval coverage for the selected dataset. It does not independently measure answer correctness or generalize to every possible repository question.

## 🔐 Environment and Data

- Repository cloning happens locally.
- Code chunks, metadata, and embeddings are stored in PostgreSQL.
- The embedding model runs locally through Sentence Transformers.
- Groq is used for LLM inference, so retrieved context is sent to the configured Groq service for answer generation.
- The `.env` file stores API credentials and database connection details and must remain private.

## 🔮 Future Improvements

- Improve retrieval accuracy through larger evaluation datasets.
- Add support for more programming languages.
- Improve source-level citations with line references.
- Experiment with hybrid retrieval and reranking.
- Improve handling of large repositories.
- Add automated tests for the complete RAG pipeline.
- Benchmark retrieval performance on larger datasets.

## 🎯 Project Objective

The main objective of this project is to understand and implement a practical RAG application for software development.

It brings together repository ingestion, code chunking, embedding generation, PostgreSQL vector storage, semantic retrieval, context management, and LLM-based answer generation to make codebase exploration easier.

---

**Built with Python, LangChain, PostgreSQL, pgvector, HuggingFace, Groq and Streamlit.**
