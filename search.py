import numpy as np

from embeddings import create_embedding


DOCUMENT_PATH = "data/documents.txt"
EMBEDDINGS_PATH = "data/embeddings.npy"


def load_documents():
    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    documents = [
        document.strip()
        for document in text.split("\n\n")
        if document.strip()
    ]

    return documents


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def search(query, documents, document_embeddings, top_k=3):

    # Convert the user's query into an embedding
    query_embedding = create_embedding(query)

    scores = []

    # Compare the query with every document
    for index, document_embedding in enumerate(document_embeddings):

        similarity = cosine_similarity(
            query_embedding,
            document_embedding
        )

        scores.append((similarity, index))

    # Sort from highest similarity to lowest
    scores.sort(reverse=True)

    return scores[:top_k]


def main():

    # Load documents
    documents = load_documents()

    # Load pre-created embeddings
    document_embeddings = np.load(EMBEDDINGS_PATH)

    print(f"Loaded {len(documents)} documents.")
    print(f"Loaded embeddings with shape: {document_embeddings.shape}")

    while True:

        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        results = search(
            query,
            documents,
            document_embeddings
        )

        print("\n--- Search Results ---\n")

        for rank, (score, index) in enumerate(results, start=1):

            print(f"{rank}. {documents[index]}")
            print(f"Similarity: {score:.4f}\n")


if __name__ == "__main__":
    main()