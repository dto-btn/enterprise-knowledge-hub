"""run_metrics table model"""
from datetime import datetime

from peewee import AutoField, ForeignKeyField, Model
from playhouse.postgres_ext import JSONField

from repository.base_model import TimestampTZField
from repository.database import db
from repository.run_history_model import RunHistory

RUN_METRICS_TABLE_NAME = "run_metrics"


class RunMetric(Model):  # pylint: disable=too-many-instance-attributes
    """run_metrics table model"""
    id: int = AutoField()
    run_history = ForeignKeyField(RunHistory, backref="metrics", column_name="run_history_id")
    metadata = JSONField(null=True)
    timestamp: datetime = TimestampTZField()

    class Meta:  # pylint: disable=too-few-public-methods
        """Configuration for the model"""
        database = db
        db_table = RUN_METRICS_TABLE_NAME