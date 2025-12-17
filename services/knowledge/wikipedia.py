import bz2
from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import re

from dotenv import load_dotenv
from services.knowledge.base import KnowledgeService

load_dotenv()

DONE_SUFFIX: str = ".done"
INDEX_FILENAME = re.compile(r"(?P<prefix>.+)-index\.txt\.bz2")

@dataclass
class WikipediaKnowedgeService(KnowledgeService):
    """Knowledge service for Wikipedia ingestion."""

    _ignored_title_prefixes: tuple[str, ...] = (
            "Draft:",
            "Category:",
            "File:",
            "Wikipedia:",
            "Ébauche:",
            "Catégorie:",
            "Fichier:",
            "Wikipédia:",
            "Portal:"
        ),

    _content_folder_path: Path = Path(os.getenv("WIKIPEDIA_CONTENT_FOLDER",
                                                 "./content/wikipedia")).expanduser().resolve()

    def __init__(self, queue_service, logger):
        super().__init__(queue_service=queue_service, logger=logger, service_name="wikipedia")

    def read(self) -> Iterator[dict[str, object]]:
        """Read data from Wikipedia index.txt.bz2 source.
        
            The content will be first entrypoint in the main .bz2 multistream file.
                Ex: 345,6789,Fruits (Read more on the doc from the README.md from content/ folder)
        """
        # Placeholder implementation for reading from Wikipedia
        for index_path in self._discover_index_files():
            match = INDEX_FILENAME.match(index_path.name)
            if not match:
                self.logger.warning("Skipping index file with unknown pattern: %s", index_path.name)
                return None
            #TODO: handle case where file wasn't fully processed before a abort/restart for instance
            self.logger.info("Reading data from Wikipedia source: %s", index_path)
            # Open up the index archive, and read line until you end up with a different byte offset (about 100 lines)
            # unzip the bites and for each articles (<page> items found in) send to the ingest queue
            last_byteoffset: int | None = None
            with bz2.open(index_path, mode='rt') as index_file:
                for line in index_file:
                    try:
                        offset_str, _, _ = line.strip().split(":", 2)
                        offset = int(offset_str)
                        if last_byteoffset is None:
                            last_byteoffset = offset
                            lenght = offset
                        else:
                            lenght = offset - last_byteoffset
                        if last_byteoffset != offset:
                            # if the byteoffset is different here it means we hit a multistream chunk end,
                            # we need to extract it (or die trying), and yield the resulting individual <page> elements
                            prefix = match.group("prefix")
                            dump_name = f"{prefix}.xml.bz2"
                            dump_path = index_path.with_name(dump_name)
                            with open(dump_path, 'rb') as dump_file:
                                dump_file.seek(last_byteoffset)
                                data = dump_file.read(lenght)
                                try:
                                    decompressed = bz2.decompress(data)
                                    xml_content = decompressed.decode("utf-8", errors="ignore")
                                    for page_match in re.finditer(r"<page>(.*?)</page>", xml_content, re.DOTALL):
                                        page_xml = page_match.group(0)
                                        yield {"wiki_page_xml": page_xml}
                                except Exception as exc:
                                    self.logger.error(
                                        "Failed to decompress chunk from %s between offsets %s and %s: %s",
                                        dump_name, last_byteoffset, offset, exc
                                    )

                            #decompressed = bz2.decompress(data)
                            #yield decompressed.decode("utf-8", errors="ignore")
                    except ValueError:
                        self.logger.warning(
                            "Skipping malformed line in %s", index_path.name
                        )

    def _discover_index_files(self) -> Iterator[Path]:
        """
        index files will be named like so for wikipedia:
            * enwiki-20240620-pages-articles-multistream1-index.txt.bz2
            * frwiki-20240620-pages-articles-multistream1-index.txt.bz2.done
        """
        self.logger.debug("Searching for index files in ---> %s", self._content_folder_path)
        for node in sorted(self._content_folder_path.rglob("*.txt*")):
            if not node.is_file():
                continue
            if node.suffix == DONE_SUFFIX:
                continue
            yield node

    def _should_ignore_title(self, title: str) -> bool:
        for prefix in self._ignored_title_prefixes:
            if title.startswith(prefix):
                return True
        return False
