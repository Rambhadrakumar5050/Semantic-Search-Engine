from chunking import chunk_document


text = """
Machine learning allows computers to learn patterns from data
and make predictions without being explicitly programmed.

Deep learning is a subset of machine learning that uses neural
networks with multiple layers to learn complex patterns.

Natural language processing allows computers to understand,
process, and generate human language.
"""


chunks = chunk_document(
    text,
    chunk_size=10,
    overlap=3
)


print("Number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):

    print(f"\nChunk {i}:")
    print(chunk)