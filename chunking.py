def chunk_document(text, chunk_size=200, overlap=50):
    """
    Split a document into smaller chunks.

    chunk_size:
        Maximum number of words in each chunk.

    overlap:
        Number of words shared between neighboring chunks.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks