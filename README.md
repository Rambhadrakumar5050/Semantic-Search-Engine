# Semantic Search Engine

An AI-powered semantic search engine built from scratch using sentence embeddings, vector similarity search, FAISS, FastAPI, Docker, and a web interface. 

Unlike traditional keyword search, this system searches for results based on the **meaning of the query**, not just exact keyword matches.

---

## What does this project do?

The user enters a natural-language query such as:

> How do computers learn from data ?

The system:

1. Converts the query into a numerical vector using a sentence-transformer model.
2. Searches a FAISS vector index containing document embeddings.
3. Calculates semantic similarity between the query and stored chunks.
4. Returns the most relevant results.
5. Displays the results through a web interface.

This allows the system to find semantically related information even when the query does not contain the exact words used in the document.

---

## Architecture

```text
                    User
                      |
                      v
              Web Search Interface
                      |
                      v
                 FastAPI API
                      |
                      v
               Search Engine
                      |
              +-------+-------+
              |               |
              v               v
       Sentence Transformer  FAISS
              |               |
              v               v
       Query Embedding   Vector Search
                              |
                              v
                        Top-K Results
                              |
                              v
                         Web Interface
