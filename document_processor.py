from pathlib import Path

from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb

VECTOR_DB = "./vector_db"

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(
    path=VECTOR_DB
)

collection = client.get_or_create_collection(
    name="knowledge_base"
)

def load_document(file_path):

    ext = Path(file_path).suffix

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    return loader.load()

def process_document(file_path):

    docs = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    texts = []

    metadata = []

    ids = []

    for i, chunk in enumerate(chunks):

        text = chunk.page_content.strip()

        texts.append(text)

        metadata.append(chunk.metadata)

        ids.append(str(i))

    embeddings = model.encode(
        texts
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadata
    )

    print(
        f"{len(chunks)} chunks stored."
    )
    print(f"Documents Loaded: {len(docs)}")
# print(f"Chunks Generated: {len(chunk)}")
# print(f"Embedding Dimension: {len(embeddings[0])}")


if __name__ == "__main__":
    process_document(
        "datasets/sample.txt"
    )