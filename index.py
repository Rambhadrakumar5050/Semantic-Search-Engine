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


def create_document_embeddings(documents):
    embeddings = []

    for document in documents:
        embedding = create_embedding(document)
        embeddings.append(embedding)

    return np.array(embeddings)


def main():
    documents = load_documents()

    print(f"Loaded {len(documents)} documents.")

    print("Creating document embeddings...")

    embeddings = create_document_embeddings(documents)

    np.save(EMBEDDINGS_PATH, embeddings)

    print("Embeddings saved!")
    print("Shape:", embeddings.shape)


if __name__ == "__main__":
    main()