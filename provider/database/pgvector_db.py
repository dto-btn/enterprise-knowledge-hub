"""
PgVector implementation of database provider
"""
import json
from typing import List, Tuple, Any
import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
from pgvector import Vector
from provider.database.base import VectorDatabaseProvider


class PgVectorProvider(VectorDatabaseProvider):
    """
    Pgvector implementation
    """

    def __init__(
        self,
        logger,
        dsn: str,
        table: str = "documents",
        embedding_dim: int = 1024,
        ivfflat_lists: int = 100
    ):
        super().__init__(logger)
        self._dsn = dsn
        self._table = table
        self._dim = embedding_dim
        self._ivfflat_lists = ivfflat_lists

        self._conn: Connection = psycopg.connect(self._dsn, row_factory=dict_row)

        register_vector(self._conn)

    def init_schema(self) -> None:
        """
        Docstring for init_schema

        :param self: Description
        """
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            cur.execute(f"""
                CREATE TABLE {self._table} (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    embedding VECTOR({self._dim}),
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
            """)

        self._conn.commit()

    def upsert(self, records, *, batch_size = 256):
        # return super().upsert(records, batch_size=batch_size)

        query = f"""
            INSERT INTO {self._table} (id, content, embedding, metadata)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata;
        """

        total = 0

        batch: List[Tuple[Any, ...]] = []

        for r in records:
            batch.append((r.id, r.text, Vector(r.embedding), json.dumps(r.metadata)))
            if len(batch) >= batch_size:
                total += self._execute_batch(query, batch)
                batch.clear()

        if batch:
            total += self._execute_batch(query, batch)

        return total

    def _execute_batch(self, sql: str, params: List[Tuple[Any, ...]]) -> int:
        with self._conn.cursor() as cur:
            cur.executemany(sql, params)
        self._conn.commit()
        return len(params)
