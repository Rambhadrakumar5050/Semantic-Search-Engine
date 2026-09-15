import numpy as np
import faiss


EMBEDDINGS_PATH = "data/chunk_embeddings.npy"
INDEX_PATH = "data/faiss_chunks.index"


def main():

    # Load chunk embeddings
    embeddings = np.load(EMBEDDINGS_PATH)

    print("Original embeddings shape:", embeddings.shape)

    # FAISS works with float32
    embeddings = embeddings.astype("float32")

    # Normalize vectors
    # After normalization, inner product becomes cosine similarity
    faiss.normalize_L2(embeddings)

    print("Embeddings normalized.")

    # Get the embedding dimension
    dimension = embeddings.shape[1]

    print("Vector dimension:", dimension)

    # Create FAISS index using Inner Product
    index = faiss.IndexFlatIP(dimension)

    # Add chunk embeddings to the index
    index.add(embeddings)

    print("Number of vectors in index:", index.ntotal)

    # Save FAISS index
    faiss.write_index(
        index,
        INDEX_PATH
    )

    print(f"FAISS chunk index saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()