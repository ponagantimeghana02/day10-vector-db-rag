from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "Artificial Intelligence helps automate tasks.",
    "Machine learning is a subset of AI.",
    "Vector databases store embeddings for fast search.",
    "Cloud computing provides scalable infrastructure."
]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("\nGenerating embeddings...\n")
document_embeddings = model.encode(documents)

vector_store = []

for doc, embedding in zip(documents, document_embeddings):
    vector_store.append({
        "document": doc,
        "embedding": embedding
    })

print("Vector Dimensions:")
print(document_embeddings.shape)
print(f"Number of documents: {document_embeddings.shape[0]}")
print(f"Embedding size: {document_embeddings.shape[1]}")

query = "How do AI systems store knowledge?"

query_embedding = model.encode([query])

similarities = cosine_similarity(
    query_embedding,
    document_embeddings
)[0]

# Find Best Match
best_index = np.argmax(similarities)

print("\nQuery:")
print(query)

print("\nSimilarity Scores:")
for doc, score in zip(documents, similarities):
    print(f"{score:.4f} -> {doc}")

print("\nTop Result:")
print(documents[best_index])

print("\nTop Similarity Score:")
print(f"{similarities[best_index]:.4f}")