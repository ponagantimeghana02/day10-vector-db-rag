import chromadb
from sentence_transformers import SentenceTransformer

documents = [
    "Artificial Intelligence helps automate tasks.",
    "Machine learning is a subset of AI.",
    "Vector databases store embeddings for fast search.",
    "Cloud computing provides scalable infrastructure."
]

metadatas = [
    {"document": "AI_Guide.pdf", "page": 1, "topic": "Artificial Intelligence"},
    {"document": "AI_Guide.pdf", "page": 2, "topic": "Machine Learning"},
    {"document": "AI_Guide.pdf", "page": 5, "topic": "Embeddings"},
    {"document": "Cloud_Guide.pdf", "page": 3, "topic": "Cloud Computing"}
]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents).tolist()

print(f"Generated {len(embeddings)} embeddings")
print(f"Embedding Dimension: {len(embeddings[0])}")

client = chromadb.Client()

collection = client.get_or_create_collection(
    name="knowledge_base"
)


collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("\nDocuments added successfully!")


query = "How are embeddings stored for semantic search?"

print(f"\nQuery: {query}")

query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\nTop Matching Results:")
print("-" * 50)

for i in range(len(results["documents"][0])):
    print(f"\nRank {i+1}")

    print("Document Chunk:")
    print(results["documents"][0][i])

    print("\nMetadata:")
    print(results["metadatas"][0][i])

    if "distances" in results:
        print(f"\nDistance Score: {results['distances'][0][i]}")