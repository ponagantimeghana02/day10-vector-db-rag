
import chromadb

from sentence_transformers import (
    SentenceTransformer
)

from config import (
    EMBEDDING_MODEL,
    CHROMA_PATH,
    COLLECTION_NAME
)


embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)



client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)


def generate_embedding(text):

    embedding = embedding_model.encode(
        text
    )

    return embedding.tolist()



def store_chunks(chunks):

    ids = []

    documents = []

    embeddings = []

    metadatas = []

    for index, chunk in enumerate(chunks):

        text = chunk.page_content

        metadata = chunk.metadata

        embedding = generate_embedding(
            text
        )

        chunk_id = (
            f"{metadata.get('source','doc')}"
            f"_{index}"
        )

        ids.append(chunk_id)

        documents.append(text)

        embeddings.append(embedding)

        metadatas.append(metadata)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(
        f"Stored {len(ids)} chunks."
    )



def document_count():

    count = collection.count()

    return count




def reset_database():

    try:

        client.delete_collection(
            COLLECTION_NAME
        )

        print(
            "Collection deleted."
        )

    except Exception as e:

        print(
            "Error:",
            e
        )



def get_collection():

    return collection



def show_documents(limit=5):

    data = collection.get()

    docs = data["documents"]

    print("\nStored Documents")

    print("-" * 50)

    for doc in docs[:limit]:

        print(doc[:200])

        print("-" * 50)




if __name__ == "__main__":

    print(
        "Collection:",
        COLLECTION_NAME
    )

    print(
        "Documents:",
        document_count()
    )

    show_documents()