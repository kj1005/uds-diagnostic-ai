"""Retrieve relevant local knowledge-base chunks without generating an answer."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

from backend.ingestion import (
    KnowledgeBaseError,
    RAGConfig,
    load_embedding_model,
)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_text: str
    section: str
    distance: float
    similarity: float
    source_filename: str
    chunk_id: str


class RAGRetriever:
    """Reusable query interface over the persistent ChromaDB knowledge base."""

    def __init__(self, config: RAGConfig | None = None) -> None:
        self.config = config or RAGConfig.load()
        try:
            import chromadb
        except ImportError as exc:
            raise KnowledgeBaseError(
                "ChromaDB is not installed. Install the project dependencies from "
                "requirements.txt."
            ) from exc

        try:
            client = chromadb.PersistentClient(
                path=str(self.config.chroma_persist_directory)
            )
            self.collection = client.get_collection(name=self.config.collection_name)
        except Exception as exc:
            raise KnowledgeBaseError(
                f"Could not open ChromaDB collection '{self.config.collection_name}' "
                f"at {self.config.chroma_persist_directory}. Run "
                "'python -m backend.ingestion' first. Details: {exc}"
            ) from exc

        self.model = load_embedding_model(self.config.embedding_model)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        """Return the nearest chunks, ordered by ascending cosine distance."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        try:
            query_embedding = self.model.encode(
                [query.strip()],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )[0]
            result: dict[str, Any] = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise KnowledgeBaseError(f"Could not retrieve knowledge-base chunks: {exc}") from exc

        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        retrieved: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if metadata is None:
                raise KnowledgeBaseError(
                    "A retrieved ChromaDB chunk is missing its required metadata."
                )
            distance_value = float(distance)
            retrieved.append(
                RetrievedChunk(
                    chunk_text=document,
                    section=str(metadata["section"]),
                    distance=distance_value,
                    similarity=1.0 - distance_value,
                    source_filename=str(metadata["source_filename"]),
                    chunk_id=str(metadata["chunk_id"]),
                )
            )
        return retrieved


def retrieve_chunks(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Convenience function for one-off retrieval; reuse RAGRetriever for repeated calls."""
    return RAGRetriever().retrieve(query, top_k=top_k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve relevant chunks from the locally indexed UDS knowledge base."
    )
    parser.add_argument("query", help="Natural-language question or search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of relevant chunks to return (default: 5).",
    )
    args = parser.parse_args()

    try:
        results = RAGRetriever().retrieve(args.query, top_k=args.top_k)
    except (KnowledgeBaseError, ValueError) as exc:
        parser.exit(status=1, message=f"Retrieval failed: {exc}\n")
    print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
