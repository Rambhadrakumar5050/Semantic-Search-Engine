import numpy as np
import faiss

from embeddings import create_embedding


DOCUMENT_PATH = "data/documents.txt"
INDEX_PATH = "data/faiss.index"


def load_documents():

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    documents = [
        document.strip()
        for document in text.split("\n\n")
        if document.strip()
    ]

    return documents


def search(query, index, documents, top_k=3):

    # Convert the user's query into an embedding
    query_embedding = create_embedding(query)

    # Convert to float32
    query_embedding = np.array(
        [query_embedding],
        dtype="float32"
    )

    # Normalize the query vector
    faiss.normalize_L2(query_embedding)

    # Search the FAISS index
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0]
    ):

        document = documents[index_position]

        results.append(
            (document, score, index_position)
        )

    return results


def main():

    # Load documents
    documents = load_documents()

    # Load FAISS index
    index = faiss.read_index(INDEX_PATH)

    print(f"Loaded {len(documents)} documents.")
    print(f"Loaded FAISS index with {index.ntotal} vectors.")

    while True:

        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        results = search(
            query,
            index,
            documents
        )

        print("\n--- Search Results ---\n")

        for rank, (document, score, index_position) in enumerate(
            results,
            start=1
        ):

            print(f"{rank}. {document}")
            print(f"Cosine similarity: {score:.4f}")
            print(f"Document index: {index_position}\n")


if __name__ == "__main__":
    main()