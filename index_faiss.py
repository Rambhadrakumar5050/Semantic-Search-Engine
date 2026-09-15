import numpy as np
import faiss


EMBEDDINGS_PATH = "data/embeddings.npy"
INDEX_PATH = "data/faiss.index"


def main():

    # Load document embeddings
    embeddings = np.load(EMBEDDINGS_PATH)

    print("Original embeddings shape:", embeddings.shape)

    # FAISS works with float32
    embeddings = embeddings.astype("float32")

    # Normalize the document vectors
    faiss.normalize_L2(embeddings)

    print("Embeddings normalized.")

    # Number of dimensions
    dimension = embeddings.shape[1]

    # Create FAISS index using Inner Product
    index = faiss.IndexFlatIP(dimension)

    # Add normalized vectors to the index
    index.add(embeddings)

    print("Number of vectors in index:", index.ntotal)

    # Save the FAISS index
    faiss.write_index(index, INDEX_PATH)

    print("FAISS cosine-similarity index saved!")


if __name__ == "__main__":
    main()