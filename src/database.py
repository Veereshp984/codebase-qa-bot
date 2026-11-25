
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()


def get_connection():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is missing from .env")

    connection = psycopg.connect(database_url)
    register_vector(connection)

    return connection


def test_connection():
    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT version()")
            print("PostgreSQL:", cursor.fetchone()[0])

            cursor.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            result = cursor.fetchone()

            if result:
                print("pgvector version:", result[0])
            else:
                print("pgvector extension not found!")

    finally:
        connection.close()


if __name__ == "__main__":
    test_connection()
