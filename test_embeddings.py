from embeddings import create_embedding


text = "Machine learning allows computers to learn from data."

vector = create_embedding(text)

print("Text:")
print(text)

print("\nVector:")
print(vector)

print("\nVector dimensions:")
print(len(vector))