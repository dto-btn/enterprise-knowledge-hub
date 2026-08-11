"""Initial migrations"""
from peewee import PostgresqlDatabase
from repository.kb_source_registry_model import KbSourceRegistry
from repository.knowledge_wikipedia_model import KnowledgeBaseWikipedia
from repository.knowledge_tbs_policies_model import KnowledgeBaseTBSPolicies
from repository.run_history_model import RunHistory
from repository.run_metrics_model import RunMetrics

def run_init_migration(db: PostgresqlDatabase):
    """Initial migrations"""
    db.connect()
    db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")
    db.create_tables([KnowledgeBaseWikipedia, KnowledgeBaseTBSPolicies, RunHistory,
                      RunMetrics, KbSourceRegistry], safe=True)
    db.close()
