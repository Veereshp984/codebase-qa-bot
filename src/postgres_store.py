
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pgvector import Vector
from psycopg.types.json import Jsonb

from src.database import get_connection


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def store_chunks(chunks, repo_id, replace_repo=False):
    """
    Generate embeddings and store code chunks in PostgreSQL.

    replace_repo=True deletes existing chunks for this repository
    before inserting the new chunks.
    """

    if not chunks:
        print("No chunks to store.")
        return

    embedding_model = get_embedding_model()

    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)

    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            if replace_repo:
                cursor.execute(
                    "DELETE FROM code_chunks WHERE repo_id = %s",
                    (repo_id,)
                )

            for chunk, embedding in zip(chunks, embeddings):
                cursor.execute(
                    """
                    INSERT INTO code_chunks
                    (content, metadata, embedding, repo_id)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (
                        chunk.page_content,
                        Jsonb(chunk.metadata),
                        Vector(embedding),
                        repo_id
                    )
                )

        connection.commit()
        print(f"Successfully stored {len(chunks)} chunks!")

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_chunk_count(repo_id):
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT COUNT(*) FROM code_chunks WHERE repo_id = %s",
                (repo_id,)
            )
            return cursor.fetchone()[0]

    finally:
        connection.close()


def retrieve_chunks(query, repo_id, k=4, language=None):
    embedding_model = get_embedding_model()

    # Convert the user's question into a 384-dimensional vector
    query_embedding = embedding_model.embed_query(query)

    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            sql = """
                SELECT content, metadata,
                       embedding <=> %s AS distance
                FROM code_chunks
                WHERE repo_id = %s
            """

            params = [Vector(query_embedding), repo_id]

            if language:
                sql += " AND metadata->>'language' = %s"
                params.append(language)

            sql += """
                ORDER BY embedding <=> %s
                LIMIT %s
            """

            params.extend([Vector(query_embedding), k])

            cursor.execute(sql, params)
            results = cursor.fetchall()

            return results

    finally:
        connection.close()


def get_neighbouring_chunks(retrieved_chunks, repo_id, window=1):
    connection = get_connection()

    expanded_chunks = {}

    try:
        with connection.cursor() as cursor:

            for content, metadata, distance in retrieved_chunks:
                source = metadata["source"]
                chunk_index = int(metadata["chunk_index"])

                start_index = max(0, chunk_index - window)
                end_index = chunk_index + window

                cursor.execute(
                    """
                    SELECT content, metadata
                    FROM code_chunks
                    WHERE repo_id = %s
                    AND metadata->>'source' = %s
                    AND (metadata->>'chunk_index')::int
                        BETWEEN %s AND %s
                    ORDER BY (metadata->>'chunk_index')::int
                    """,
                    (
                        repo_id,
                        source,
                        start_index,
                        end_index
                    )
                )

                results = cursor.fetchall()

                for chunk_content, chunk_metadata in results:
                    key = (
                        source,
                        int(chunk_metadata["chunk_index"])
                    )

                    expanded_chunks[key] = Document(
                        page_content=chunk_content,
                        metadata=chunk_metadata
                    )

        return [
            document
            for _, document in sorted(expanded_chunks.items())
        ]

    finally:
        connection.close()


def explain_retrieval(query, repo_id, k=4):
    embedding_model = get_embedding_model()
    query_embedding = embedding_model.embed_query(query)

    connection = get_connection()

    try:
        with connection.cursor() as cursor:

            # Diagnostic only: discourage sequential scans
            cursor.execute("SET enable_seqscan = OFF")

            cursor.execute(
                """
                EXPLAIN ANALYZE
                SELECT content, metadata,
                       embedding <=> %s AS distance
                FROM code_chunks
                WHERE repo_id = %s
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (
                    Vector(query_embedding),
                    repo_id,
                    Vector(query_embedding),
                    k
                )
            )

            for row in cursor.fetchall():
                print(row[0])

    finally:
        connection.close()
