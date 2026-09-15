import json
import numpy as np
import faiss

from embeddings import create_embedding


CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "data/faiss_chunks.index"

# Minimum similarity required for a result
MIN_SIMILARITY = 0.40


def load_chunks():

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


# Load chunks and FAISS index once
chunks = load_chunks()
index = faiss.read_index(INDEX_PATH)


def search(query, top_k=3):

    # Make sure top_k is an integer
    top_k = int(top_k)

    # Prevent invalid values
    if top_k < 1:
        top_k = 1

    # Never request more results than we have chunks
    top_k = min(top_k, len(chunks))

    # Convert the query into an embedding
    query_embedding = create_embedding(query)

    # Convert to float32
    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    # Normalize the query vector
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, chunk_index in zip(
        scores[0],
        indices[0]
    ):

        # Ignore results below our similarity threshold
        if score < MIN_SIMILARITY:
            continue

        chunk = chunks[chunk_index]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "source": chunk["source"],
                "text": chunk["text"],
                "score": float(score)
            }
        )

    return results