import json

from chunking import chunk_document


DOCUMENT_PATH = "data/documents.txt"
METADATA_PATH = "data/document_metadata.json"
CHUNKS_PATH = "data/chunks.json"


def load_documents():

    with open(DOCUMENT_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    documents = [
        document.strip()
        for document in text.split("\n\n")
        if document.strip()
    ]

    return documents


def load_metadata():

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def main():

    documents = load_documents()
    metadata = load_metadata()

    all_chunks = []

    chunk_id = 0

    for document_id, document in enumerate(documents):

        document_metadata = metadata[document_id]

        chunks = chunk_document(
            document,
            chunk_size=200,
            overlap=50
        )

        for chunk in chunks:

            chunk_data = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title": document_metadata["title"],
                "source": document_metadata["source"],
                "text": chunk
            }

            all_chunks.append(chunk_data)

            chunk_id += 1

    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:

        json.dump(
            all_chunks,
            file,
            indent=4
        )

    print(f"Created {len(all_chunks)} chunks.")
    print(f"Saved chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()