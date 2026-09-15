import json
import numpy as np

from embeddings import create_embedding


CHUNKS_PATH = "data/chunks.json"
EMBEDDINGS_PATH = "data/chunk_embeddings.npy"


def load_chunks():

    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    return chunks


def create_chunk_embeddings(chunks):

    embeddings = []

    for chunk in chunks:

        text = chunk["text"]

        embedding = create_embedding(text)

        embeddings.append(embedding)

    return np.array(embeddings)


def main():

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks.")

    print("Creating chunk embeddings...")

    embeddings = create_chunk_embeddings(chunks)

    print("Embeddings created.")

    print("Embedding shape:", embeddings.shape)

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print(f"Embeddings saved to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()