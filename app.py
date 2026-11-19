
import streamlit as st
from pathlib import Path
from urllib.parse import urlparse

from src.clone_repo import clone_repo
from src.build_store import process_repository
from src.retrieve import (
    load_vector_store,
    retrieve_chunks,
    get_neighbouring_chunks
)
from src.generate_answer import (
    rewrite_question,
    generate_answer
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Codebase Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
}

section[data-testid="stSidebar"] {
    background-color: #151923;
    border-right: 1px solid #292f3b;
}

.main-title {
    font-size: 30px;
    font-weight: 700;
    color: #f1f5f9;
}

.sub-title {
    color: #94a3b8;
    font-size: 14px;
}

.panel-title {
    font-size: 19px;
    font-weight: 600;
    margin-bottom: 12px;
}

.repo-status {
    background-color: #172b25;
    padding: 12px;
    border-radius: 8px;
    color: #4ade80;
    font-size: 13px;
}

div[data-testid="stChatMessage"] {
    border: 1px solid #252b36;
    border-radius: 12px;
    padding: 12px;
    background-color: #151923;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

defaults = {
    "repo_id": None,
    "vector_store": None,
    "chat_history": [],
    "repo_path": None,
    "latest_chunks": [],
    "latest_sources": []
}

for key, value in defaults.items():

    if key not in st.session_state:
        st.session_state[key] = value


# --------------------------------------------------
# REPOSITORY URL VALIDATION
# --------------------------------------------------

def get_repo_details(repo_url):

    parsed_url = urlparse(repo_url)

    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc != "github.com"
    ):
        raise ValueError("Please enter a valid GitHub URL.")

    parts = parsed_url.path.strip("/").split("/")

    if len(parts) != 2 or not all(parts):
        raise ValueError("Invalid GitHub repository URL.")

    owner, repo_name = parts

    repo_name = repo_name.removesuffix(".git")

    repo_id = f"{owner}_{repo_name}"

    return owner, repo_name, repo_id


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.markdown("## 🤖 Codebase Q&A")

    st.caption("AI-powered repository assistant")

    st.divider()

    st.markdown("### Connect Repository")

    with st.form("repository_form"):

        repo_url = st.text_input(
            "GitHub Repository URL",
            placeholder="https://github.com/user/repository"
        )

        connect = st.form_submit_button(
            "＋ Clone and Index",
            use_container_width=True,
            type="primary"
        )

    if connect:

        if not repo_url.strip():

            st.error("Enter a GitHub repository URL.")

        else:

            try:

                owner, repo_name, repo_id = get_repo_details(repo_url)

                with st.status(
                    "Connecting repository...",
                    expanded=True
                ) as status:

                    st.write("Cloning repository...")

                    repo_path = clone_repo(repo_url)

                    if repo_path is None:
                        raise RuntimeError("Repository cloning failed.")

                    st.write("Indexing repository files...")

                    process_repository(
                        repo_path=repo_path,
                        repo_id=repo_id,
                        rebuild=False
                    )

                    st.write("Loading vector database...")

                    vector_store = load_vector_store(repo_id)

                    status.update(
                        label="Repository connected successfully!",
                        state="complete",
                        expanded=False
                    )

                st.session_state.repo_id = repo_id
                st.session_state.vector_store = vector_store
                st.session_state.repo_path = str(repo_path)

                st.session_state.chat_history = []
                st.session_state.latest_chunks = []
                st.session_state.latest_sources = []

                st.success(f"Connected: {repo_name}")

            except Exception as error:

                st.error(f"Connection failed: {error}")

    st.divider()

    # Current repository

    st.markdown("### Current Repository")

    if st.session_state.repo_id:

        st.markdown(
            f"""
            <div class="repo-status">
                🟢 Connected<br><br>
                <b>{st.session_state.repo_id}</b>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.info("No repository connected.")

    st.divider()

    # Chat controls

    st.markdown("### Chat History")

    if st.button(
        "＋ New Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []
        st.session_state.latest_chunks = []
        st.session_state.latest_sources = []

        st.rerun()

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            st.caption(f"💬 {message['content'][:35]}")

    st.divider()

    st.caption("Powered by LangChain, ChromaDB and Groq")


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">Codebase Q&A Bot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Understand, explore and debug any GitHub repository using AI.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# --------------------------------------------------
# TWO-PANEL LAYOUT
# --------------------------------------------------

chat_col, source_col = st.columns(
    [1.5, 1],
    gap="large"
)


# --------------------------------------------------
# LEFT PANEL - CHAT
# --------------------------------------------------

with chat_col:

    st.markdown(
        '<div class="panel-title">💬 Ask Your Codebase</div>',
        unsafe_allow_html=True
    )

    if st.session_state.vector_store is None:

        st.info(
            "Connect a GitHub repository from the sidebar to start chatting."
        )

    else:

        # Display conversation history

        for message in st.session_state.chat_history:

            with st.chat_message(message["role"]):

                st.markdown(message["content"])

                if (
                    message["role"] == "assistant"
                    and message.get("sources")
                ):

                    with st.expander("📁 View source files"):

                        for source in message["sources"]:
                            st.code(source)

        # Chat input

        query = st.chat_input(
            "Ask something about your codebase..."
        )

        if query:

            st.session_state.chat_history.append({
                "role": "user",
                "content": query
            })

            try:

                with st.spinner("Understanding your question..."):

                    rewritten_query = rewrite_question(
                        query=query,
                        chat_history=st.session_state.chat_history[:-1]
                    )

                with st.spinner("Searching the codebase..."):

                    retrieved_chunks = retrieve_chunks(
                        vector_store=st.session_state.vector_store,
                        query=rewritten_query,
                        k=6,
                        language=None,
                        use_mmr=False
                    )

                    expanded_chunks = get_neighbouring_chunks(
                        vector_store=st.session_state.vector_store,
                        retrieved_chunks=retrieved_chunks,
                        window=1
                    )

                with st.spinner("Generating answer..."):

                    answer = generate_answer(
                        query=query,
                        retrieved_chunks=[
                            (doc, None)
                            for doc in expanded_chunks
                        ]
                    )

                source_paths = sorted({
                    doc.metadata["source"]
                    for doc in expanded_chunks
                })

                st.session_state.latest_chunks = [
                    {
                        "source": doc.metadata.get("source", ""),
                        "content": doc.page_content,
                        "chunk_index": doc.metadata.get("chunk_index", 0)
                    }
                    for doc in expanded_chunks
                ]

                st.session_state.latest_sources = source_paths

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": source_paths
                })

            except Exception as error:

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Something went wrong: {error}",
                    "sources": []
                })

            st.rerun()


# --------------------------------------------------
# RIGHT PANEL - REPOSITORY EXPLORER
# --------------------------------------------------

with source_col:

    st.markdown(
        '<div class="panel-title">📂 Repository Explorer</div>',
        unsafe_allow_html=True
    )

    if st.session_state.repo_path is None:

        st.info(
            "Connect a GitHub repository to explore its files."
        )

    else:

        repo_path = Path(st.session_state.repo_path)

        SKIP_DIRS = {
            ".git",
            "node_modules",
            "venv",
            ".venv",
            "__pycache__",
            "dist",
            "build"
        }

        # Collect repository files

        repository_files = []

        for file_path in repo_path.rglob("*"):

            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(repo_path)

            if any(
                part in SKIP_DIRS
                for part in relative_path.parts
            ):
                continue

            repository_files.append(relative_path)

        repository_files.sort(
            key=lambda path: str(path).lower()
        )

        st.caption(f"Total files: {len(repository_files)}")

        if repository_files:

            # File selection

            selected_file = st.selectbox(
                "Browse repository files",
                options=repository_files,
                format_func=lambda path: str(path),
                key="selected_repository_file"
            )

            full_file_path = repo_path / selected_file

            st.caption(f"📄 {selected_file}")

            try:

                # Read complete source file

                file_content = full_file_path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                # Programming language detection

                language_map = {
                    ".py": "python",
                    ".js": "javascript",
                    ".ts": "typescript",
                    ".tsx": "typescript",
                    ".jsx": "javascript",
                    ".java": "java",
                    ".md": "markdown",
                    ".json": "json",
                    ".html": "html",
                    ".css": "css",
                    ".sql": "sql",
                    ".yaml": "yaml",
                    ".yml": "yaml",
                    ".xml": "xml",
                    ".sh": "bash"
                }

                language = language_map.get(
                    full_file_path.suffix.lower(),
                    "text"
                )

                total_lines = len(file_content.splitlines())

                st.markdown(
                    f"**Complete source file** · {total_lines} lines"
                )

                # Display complete file

                st.code(
                    file_content,
                    language=language,
                    line_numbers=True
                )

            except OSError as error:

                st.error(f"Could not read file: {error}")

        else:

            st.info("No files found in this repository.")
