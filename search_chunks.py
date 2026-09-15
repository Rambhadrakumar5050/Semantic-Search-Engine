from search_engine import search


def main():

    print("Semantic Search Engine")
    print("Type 'exit' to quit.")

    while True:

        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        results = search(query)

        print("\n--- Search Results ---\n")

        for rank, result in enumerate(
            results,
            start=1
        ):

            print(f"Result {rank}")
            print(f"Title: {result['title']}")
            print(f"Source: {result['source']}")
            print(f"Cosine similarity: {result['score']:.4f}")
            print(f"Text: {result['text']}")
            print()


if __name__ == "__main__":
    main()