import chromadb
from sentence_transformers import SentenceTransformer

documents = [
    "Artificial Intelligence helps automate tasks.",
    "Machine learning is a subset of AI.",
    "Vector databases store embeddings for fast search.",
    "Cloud computing provides scalable infrastructure."
]

# ✅ FIXED METADATA KEY = "source"
metadatas = [
    {"source": "AI_Guide.pdf", "page": 1, "topic": "Artificial Intelligence"},
    {"source": "AI_Guide.pdf", "page": 2, "topic": "Machine Learning"},
    {"source": "AI_Guide.pdf", "page": 5, "topic": "Embeddings"},
    {"source": "Cloud_Guide.pdf", "page": 3, "topic": "Cloud Computing"}
]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(documents).tolist()

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="knowledge_base")

collection.add(
    ids=[f"doc_{i}" for i in range(len(documents))],
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("Vector DB created successfully!")