"""Persistence model for the knowledge-source embedding registry."""
from peewee import IntegerField, Model, TextField

from repository.base_model import TimestampTZField
from repository.database import db

KB_SOURCE_REGISTRY_TABLE = "kb_source_registry"


class KbSourceRegistry(Model):
    """
    One row per knowledge source slug (e.g. 'wikipedia', 'tbs-policies').
    Written/updated at the start of every ingest run so the query layer
    always knows which model and instruction to use when embedding a query.
    """
    source = TextField(primary_key=True)   # matches service_name / run-management slug
    model_name = TextField()               # e.g. "Qwen/Qwen3-Embedding-0.6B"
    dimensions = IntegerField()            # vector size actually stored (e.g. 512)
    query_instruction = TextField()        # task-specific Qwen3 instruction prefix
    last_ingested_at = TimestampTZField(null=True)

    class Meta:  # pylint: disable=too-few-public-methods
        """Configuration for the model."""
        database = db
        db_table = KB_SOURCE_REGISTRY_TABLE
