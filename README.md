# AI-Powered UDS Diagnostic Assistant

This academic MVP uses the synthetic knowledge base in
`data/documents/UDS_Synthetic_Knowledge_Base.txt` as a local retrieval source.
It does not call an external LLM or replace an official UDS specification.

## Local RAG ingestion and retrieval

Install the dependencies listed in `requirements.txt`, then from the project
root run:

```powershell
python -m backend.ingestion
```

The command parses the numbered sections, groups their text into chunks, embeds
them with `sentence-transformers/all-MiniLM-L6-v2`, and stores the chunks,
embeddings, and source metadata in the persistent ChromaDB collection configured
in `config/model_config.json` (`data/chroma_db` by default). The model is run
locally after it is available in the Hugging Face cache; its first use may
download the model files.

Query the indexed chunks from the command line:

```powershell
python -m backend.rag "What does NRC 0x31 mean for an unsupported DID?" --top-k 3
```

The command prints a JSON array containing each chunk's text, section title,
source filename, stable chunk ID, cosine distance, and cosine similarity
(`1 - distance`). Smaller distances indicate closer matches. To make repeated
queries from Python without reloading the embedding model, create one
`backend.rag.RAGRetriever` and call its `retrieve(query, top_k=5)` method.

The index is local and persistent. Re-running ingestion replaces the indexed
chunks for that source file, avoiding stale duplicates. Retrieval returns
context only; it does not generate or validate diagnostic answers.