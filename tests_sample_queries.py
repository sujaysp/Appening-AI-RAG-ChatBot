from src.graph import build_rag_graph


graph = build_rag_graph()


queries = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]


for index, query in enumerate(queries, start=1):
    print("\n" + "=" * 100)
    print(f"TEST {index}")
    print(f"QUERY: {query}")
    print("=" * 100)

    result = graph.invoke(
        {
            "question": query,
        }
    )

    print("\nFINAL ANSWER:")
    print(result.get("answer"))

    print("\nCONFIDENCE SCORE:")
    print(result.get("confidence_score"))

    print("\nRETRIEVED CHUNKS:")
    for chunk_index, context in enumerate(
        result.get("context", []),
        start=1,
    ):
        metadata = result.get("context_metadata", [])

        if chunk_index - 1 < len(metadata):
            page = metadata[chunk_index - 1].get("page")
            source = metadata[chunk_index - 1].get("source")
        else:
            page = None
            source = None

        print(f"\n--- Chunk {chunk_index} ---")
        print(f"Page: {page}")
        print(f"Source: {source}")
        print(context[:700])