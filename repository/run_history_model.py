"""run_history table model"""
from datetime import datetime

from peewee import AutoField, IntegerField, TextField, Model
from repository.base_model import TimestampTZField
from repository.database import db

RUN_HISTORY_TABLE_NAME = "run_history"

class RunHistory(Model): #pylint: disable=too-many-instance-attributes
    """run_history table model"""
    id: int = AutoField()
    run_id: int | None = IntegerField(null=True)
    service_name: str = TextField()
    status: str = TextField()
    metadata: str = TextField(null=True)
    timestamp: datetime = TimestampTZField()

    class Meta: # pylint: disable=too-few-public-methods
        """Configuration for the model"""
        database = db
        db_table = RUN_HISTORY_TABLE_NAME
