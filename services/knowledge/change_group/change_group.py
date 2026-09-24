"""Change Group knowledge service backed by PDF documents.

This is intentionally a simple first iteration: it scans a local folder that stands in
for a specific SharePoint document library, extracts text from each PDF, and pushes the
chunks through the same raw -> process -> store pipeline as the other knowledge sources.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader

from repository.knowledge_change_group_model import KnowledgeBaseChangeGroup
from services.database.change_group_service import ChangeGroupService
from services.knowledge.base import KnowledgeService
from services.knowledge.models import KnowledgeItem
from services.knowledge.change_group.models import ChangeGroupItemRaw, ChangeGroupItemProcessed


@dataclass
class ChangeGroupKnowledgeService(KnowledgeService):
    """Knowledge service for the Change Group document collection."""

    root_folder: str = os.getenv("SHAREPOINT_LOCAL_ROOT", "./content/sharepoint-demo")

    def __init__(self, queue_service, logger, run_history_service, root_folder: str | None = None, service_name: str = "change-group"):
        super().__init__(
            queue_service=queue_service,
            logger=logger,
            run_history_service=run_history_service,
            service_name=service_name,
        )
        self.root_folder = root_folder or os.getenv("SHAREPOINT_LOCAL_ROOT", "./content/sharepoint-demo")
        self._change_group_service = ChangeGroupService(logger)

    def get_query_instruction(self) -> str:
        return (
            "Instruct: Given a query, retrieve relevant Change Group document passages that answer the query\n"
            "Query: "
        )

    def get_batch_size(self) -> int:
        return int(os.getenv("SHAREPOINT_PROCESS_BATCH_SIZE", "32"))

    def _get_run_id(self) -> int:
        root = Path(self.root_folder).expanduser().resolve()
        files = sorted(root.glob("**/*.pdf")) if root.exists() else []
        if not files:
            return int(hashlib.sha256(f"change-group-{datetime.now().isoformat()}".encode()).hexdigest()[:8], 16) & 0x7FFFFFFF
        run_id_hash = hashlib.sha256()
        for file in files:
            stat = file.stat()
            run_id_hash.update(file.name.encode())
            run_id_hash.update(str(stat.st_size).encode())
            run_id_hash.update(str(stat.st_mtime_ns).encode())
        return int.from_bytes(run_id_hash.digest()[:4], "big", signed=False) & 0x7FFFFFFF

    def fetch_from_source(self) -> Iterator[ChangeGroupItemRaw]:
        root = Path(self.root_folder).expanduser().resolve()
        if not root.exists():
            self.logger.warning("SharePoint root folder does not exist: %s", root)
            return

        for pdf_path in sorted(root.rglob("*.pdf")):
            try:
                item = self._read_pdf(pdf_path)
                if item is None:
                    continue

                try:
                    if self._change_group_service.record_is_up_to_date(item.file_path, item.source, item.last_modified_date):
                        self.logger.debug("Document %s is up to date, skipping.", item.file_path)
                        continue
                    self._change_group_service.delete_by_file_source(item.file_path, item.source)
                except Exception:  # pragma: no cover - defensive against missing/uninitialized DB tables.
                    self.logger.warning(
                        "Skipping dedupe cleanup for %s because the SharePoint repository is not ready yet.",
                        item.file_path,
                    )
                yield item
            except Exception:
                self.logger.exception("Failed to process PDF %s", pdf_path)
                continue

    def _read_pdf(self, pdf_path: Path) -> ChangeGroupItemRaw | None:
        try:
            content = self.extract_text_from_pdf(pdf_path)
            if not content:
                self.logger.warning("PDF has no extractable text: %s", pdf_path)
                return None
            return ChangeGroupItemRaw(
                name=pdf_path.name,
                content=content,
                file_path=str(pdf_path),
                source=self.service_name,
                last_modified_date=datetime.fromtimestamp(pdf_path.stat().st_mtime),
            )
        except Exception:
            self.logger.exception("Failed to read PDF %s", pdf_path)
            return None

    def emit_fetched_item(self, item: ChangeGroupItemRaw) -> None:
        from provider.embedding.qwen3.embedder_factory import get_embedder  # pylint: disable=import-outside-toplevel
        embedder = get_embedder()
        max_tokens = getattr(embedder, "max_seq_length", None)
        chunks = embedder.chunk_text_by_tokens(item.content, max_tokens=max_tokens)
        num_chunks = len(chunks)

        for idx, chunk_text in enumerate(chunks, start=1):
            chunk_item = ChangeGroupItemRaw(
                name=item.name,
                content=chunk_text,
                file_path=item.file_path,
                source=item.source,
                last_modified_date=item.last_modified_date,
                chunk_index=idx,
                chunk_count=num_chunks,
            )
            self.queue_service.write(self._ingest_queue_name(), chunk_item)

    def process_item(self, knowledge_item: KnowledgeItem) -> None:
        from provider.embedding.qwen3.embedder_factory import get_embedder  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        embedder = get_embedder()
        content = knowledge_item['content'] if isinstance(knowledge_item, dict) else knowledge_item.content
        embeddings = embedder.embed([content])
        vec = np.asarray(embeddings)[0]

        item_data = knowledge_item if isinstance(knowledge_item, dict) else knowledge_item.model_dump()
        processed = ChangeGroupItemProcessed(
            name=item_data['name'],
            content=item_data['content'],
            file_path=item_data['file_path'],
            source=item_data['source'],
            last_modified_date=item_data.get('last_modified_date'),
            chunk_index=item_data.get('chunk_index', 1),
            chunk_count=item_data.get('chunk_count', 1),
            embeddings=vec,
        )
        self.emit_processed_item(processed)

    def emit_processed_item(self, item: ChangeGroupItemProcessed) -> None:
        self.queue_service.write(self._processed_queue_name(), item)

    def store_item(self, item: KnowledgeItem) -> None:
        validated = ChangeGroupItemProcessed.model_validate(item)
        record_to_insert = KnowledgeBaseChangeGroup.from_item(validated)
        self._change_group_service.insert(record_to_insert.as_mapping())

    def extract_text_from_pdf(self, pdf_path: str | Path) -> str:
        path = Path(pdf_path)
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        return "\n\n".join(pages).strip()
