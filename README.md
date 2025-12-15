# enterprise-knowledge-hub
Enterprise Knowledge Hub

## Initial setup

Make sure you add this file to the root: `.env` (refer to `.env.example`)

To start the docker container: `docker compose up -d`

### Database Setup

```bash
docker exec -it postgres-rag psql -U admin -d rag
```

On first run ensure you have this table created:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
   id SERIAL PRIMARY KEY,
   file_name TEXT NOT NULL,
   content TEXT,
   embedding VECTOR(1024),
   CONSTRAINT uniquename UNIQUE (file_name)
);
```

### Running locally

**Requires UV**, see [isntallation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
uv install
# see scraping instructions below first..
uv run uvicorn app.main:app
```
