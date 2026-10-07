"""Parse and index the local synthetic UDS knowledge base."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "model_config.json"
DEFAULT_SOURCE_PATH = (
    PROJECT_ROOT / "data" / "documents" / "UDS_Synthetic_Knowledge_Base.txt"
)
SECTION_HEADING = re.compile(
    r"(?m)^\s*(\d{1,2})\.\s+(.+?)\s*(?=\r?\n[ \t]*={5,}[ \t]*(?:\r?\n|$))"
)
DECORATIVE_LINE = re.compile(r"^\s*(?:={5,}|-{5,})\s*$")


class KnowledgeBaseError(RuntimeError):
    """Raised when the knowledge base cannot be parsed or indexed."""


@dataclass(frozen=True)
class RAGConfig:
    embedding_model: str
    chunk_size: int
    chroma_persist_directory: Path
    collection_name: str

    @classmethod
    def load(cls) -> "RAGConfig":
        try:
            raw: dict[str, Any] = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KnowledgeBaseError(f"RAG configuration not found: {CONFIG_PATH}") from exc
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseError(
                f"RAG configuration contains invalid JSON: {CONFIG_PATH}: {exc}"
            ) from exc
        except OSError as exc:
            raise KnowledgeBaseError(
                f"Could not read RAG configuration at {CONFIG_PATH}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise KnowledgeBaseError("RAG configuration must contain a JSON object.")

        model_name = raw.get("embedding_model")
        chunk_size = raw.get("chunk_size")
        persist_directory = raw.get("chroma_persist_directory")
        collection_name = raw.get("collection_name")
        if not isinstance(model_name, str) or not model_name.strip():
            raise KnowledgeBaseError("RAG config 'embedding_model' must be a non-empty string.")
        if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
            raise KnowledgeBaseError("RAG config 'chunk_size' must be a positive integer.")
        if not isinstance(persist_directory, str) or not persist_directory.strip():
            raise KnowledgeBaseError(
                "RAG config 'chroma_persist_directory' must be a non-empty string."
            )
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise KnowledgeBaseError("RAG config 'collection_name' must be a non-empty string.")

        persist_path = Path(persist_directory)
        if not persist_path.is_absolute():
            persist_path = PROJECT_ROOT / persist_path

        return cls(
            embedding_model=model_name.strip(),
            chunk_size=chunk_size,
            chroma_persist_directory=persist_path,
            collection_name=collection_name.strip(),
        )


@dataclass(frozen=True)
class KnowledgeSection:
    title: str
    text: str
    number: int


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    text: str
    section: str
    source_filename: str


def _clean_section_text(text: str) -> str:
    lines = [line for line in text.splitlines() if not DECORATIVE_LINE.fullmatch(line)]
    return "\n".join(lines).strip()


def parse_sections(document_text: str) -> list[KnowledgeSection]:
    """Split a numbered knowledge-base document into its introduction and sections."""
    if not document_text.strip():
        raise KnowledgeBaseError("The knowledge-base document is empty.")

    headings = list(SECTION_HEADING.finditer(document_text))
    if not headings:
        raise KnowledgeBaseError(
            "No numbered section headings were found in the knowledge-base document."
        )

    sections: list[KnowledgeSection] = []
    preamble = _clean_section_text(document_text[: headings[0].start()])
    if preamble:
        sections.append(
            KnowledgeSection(
                title="Knowledge Base Introduction",
                text=preamble,
                number=0,
            )
        )

    for index, heading in enumerate(headings):
        content_start = heading.end()
        content_end = headings[index + 1].start() if index + 1 < len(headings) else len(document_text)
        section_text = _clean_section_text(document_text[content_start:content_end])
        if section_text:
            sections.append(
                KnowledgeSection(
                    title=heading.group(2).strip(),
                    text=section_text,
                    number=int(heading.group(1)),
                )
            )

    if not sections:
        raise KnowledgeBaseError("No non-empty sections were found in the knowledge base.")
    return sections


def _split_long_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]
    return [
        part.strip()
        for part in textwrap.wrap(
            paragraph,
            width=chunk_size,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=False,
            drop_whitespace=True,
        )
        if part.strip()
    ]


def create_chunks(
    sections: list[KnowledgeSection],
    source_filename: str,
    chunk_size: int,
) -> list[KnowledgeChunk]:
    """Group paragraphs into bounded chunks without crossing section boundaries."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")

    chunks: list[KnowledgeChunk] = []
    for section in sections:
        paragraphs: list[str] = []
        for paragraph in re.split(r"\n\s*\n", section.text):
            paragraph = paragraph.strip()
            if paragraph:
                paragraphs.extend(_split_long_paragraph(paragraph, chunk_size))

        grouped: list[str] = []
        current = ""
        for paragraph in paragraphs:
            combined = f"{current}\n\n{paragraph}" if current else paragraph
            if current and len(combined) > chunk_size:
                grouped.append(current)
                current = paragraph
            else:
                current = combined
        if current:
            grouped.append(current)

        for chunk_index, chunk_text in enumerate(grouped, start=1):
            chunk_id = (
                f"uds-kb-s{section.number:02d}-c{chunk_index:04d}"
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    section=section.title,
                    source_filename=source_filename,
                )
            )

    if not chunks:
        raise KnowledgeBaseError("No text chunks were created from the knowledge base.")
    return chunks


def load_embedding_model(model_name: str) -> Any:
    """Load the configured local Sentence Transformer model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise KnowledgeBaseError(
            "Sentence Transformers is not installed. Install the project dependencies "
            "from requirements.txt."
        ) from exc

    try:
        model = SentenceTransformer(model_name, device="cpu")
        model.eval()
        return model
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not load embedding model '{model_name}'. Ensure it is available "
            f"locally or can be downloaded to the Hugging Face cache: {exc}"
        ) from exc


def ingest_knowledge_base(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    config: RAGConfig | None = None,
) -> int:
    """Embed and persist all chunks from the configured source document."""
    source = Path(source_path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    if not source.is_file():
        raise KnowledgeBaseError(f"Knowledge-base file not found: {source}")

    try:
        document_text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        raise KnowledgeBaseError(f"Could not read knowledge-base file {source}: {exc}") from exc

    settings = config or RAGConfig.load()
    sections = parse_sections(document_text)
    chunks = create_chunks(sections, source.name, settings.chunk_size)
    model = load_embedding_model(settings.embedding_model)
    embedding_texts = [f"Section: {chunk.section}\n\n{chunk.text}" for chunk in chunks]
    try:
        embeddings = model.encode(
            embedding_texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise KnowledgeBaseError(f"Could not generate knowledge-base embeddings: {exc}") from exc

    try:
        import chromadb
    except ImportError as exc:
        raise KnowledgeBaseError(
            "ChromaDB is not installed. Install the project dependencies from requirements.txt."
        ) from exc

    try:
        client = chromadb.PersistentClient(path=str(settings.chroma_persist_directory))
        collection = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        # Replace this source's old chunks so repeated ingestion cannot leave stale entries.
        collection.delete(where={"source_filename": source.name})
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings.tolist(),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source_filename": chunk.source_filename,
                    "section": chunk.section,
                    "chunk_id": chunk.chunk_id,
                }
                for chunk in chunks
            ],
        )
    except Exception as exc:
        raise KnowledgeBaseError(
            f"Could not persist chunks to ChromaDB at "
            f"{settings.chroma_persist_directory}: {exc}"
        ) from exc

    return len(chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest the local UDS knowledge base.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Knowledge-base text file (default: project data/documents file).",
    )
    args = parser.parse_args()

    try:
        count = ingest_knowledge_base(args.source)
    except KnowledgeBaseError as exc:
        parser.exit(status=1, message=f"Ingestion failed: {exc}\n")
    print(f"Ingested {count} chunks into the persistent local ChromaDB collection.")


if __name__ == "__main__":
    main()
