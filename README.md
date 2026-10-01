# Fieldnotes Research Desk

A local Streamlit app for indexing text documents in ChromaDB, retrieving relevant passages, and preparing source-grounded client reports.

## Run

```bash
streamlit run home.py
```

The ChromaDB collection is persisted in `chroma_db/` beside the app. On first indexing, ChromaDB may download its default local embedding model. Uploads accept any filename extension when the file contains readable text, up to 20 MB each; HTML is converted to readable text before indexing.

An OpenAI API key is optional. Set `OPENAI_API_KEY` in the environment or enter it in the app sidebar to enable AI-assisted search-query expansion and grounded answer generation. Without a key, vector retrieval still works and responses contain concise source excerpts. Review citations against the source passages before sending a report to a client.