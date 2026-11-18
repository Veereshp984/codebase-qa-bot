from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter
)

from langchain_core.documents import Document




LANGUAGE_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".java": Language.JAVA,
    ".md": Language.MARKDOWN
}





def split_documents(documents):

    all_chunks = []

    for doc in documents:

        # Get the file language
        language = doc.metadata.get("language")

        # Find the corresponding extension
        extension_map = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "java": ".java",
            "markdown": ".md"
        }

        extension = extension_map.get(language)

        # Select language-aware splitter
        if extension in LANGUAGE_MAP:

            splitter = RecursiveCharacterTextSplitter.from_language(
                language=LANGUAGE_MAP[extension],
                chunk_size=1000,
                chunk_overlap=100
            )

        else:
            # Fallback for unsupported languages
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

        # Split the document content
        text_chunks = splitter.split_text(doc.page_content)

        # Convert each chunk into a Document
        for index, chunk in enumerate(text_chunks):

            chunk_doc = Document(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    "chunk_index": index
                }
            )

            all_chunks.append(chunk_doc)

    return all_chunks
