import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="knowledge_base")

print("Semantic Document Search Ready!")
print("-" * 50)

while True:
    query = input("\nEnter your question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    print("\nTop Matching Results")
    print("=" * 50)

    for i in range(len(results["documents"][0])):

        metadata = results["metadatas"][0][i]

        # ✅ FIX HERE (NO MORE UNKNOWN)
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "N/A")
        topic = metadata.get("topic", "N/A")

        print(f"\nResult #{i+1}")

        print(f"Document Name : {source}")
        print(f"Page          : {page}")
        print(f"Topic         : {topic}")

        print(f"Similarity Score : {1 - results['distances'][0][i]:.4f}")

        print("\nRelevant Text Chunk:")
        print(results["documents"][0][i])
        print("-" * 50)