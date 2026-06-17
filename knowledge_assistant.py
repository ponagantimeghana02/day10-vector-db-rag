import os
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader
)

# from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq

# -----------------------------------
# Load Environment Variables
# -----------------------------------
load_dotenv()

# -----------------------------------
# LLM
# -----------------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

# -----------------------------------
# Embedding Model
# -----------------------------------
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# -----------------------------------
# ChromaDB
# -----------------------------------
client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="knowledge_base"
)

# -----------------------------------
# Upload + Index Documents
# -----------------------------------
def load_document(file_path):

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith(".txt"):
        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

    else:
        raise ValueError(
            "Only PDF and TXT files supported"
        )

    return loader.load()


def index_document(file_path):

    docs = load_document(file_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):

        text = chunk.page_content

        embedding = embedding_model.encode(
            text
        ).tolist()

        collection.add(
            ids=[f"{os.path.basename(file_path)}_{i}"],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{
                "source": file_path
            }]
        )

    print(
        f"Indexed {len(chunks)} chunks "
        f"from {file_path}"
    )

# -----------------------------------
# Retriever
# -----------------------------------
def retrieve(question, k=3):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return list(zip(docs, metas))

# -----------------------------------
# Prompt
# -----------------------------------
prompt = ChatPromptTemplate.from_template("""
You are a Knowledge Base Assistant.

Use ONLY the provided context.

If answer is not found, say:
"I don't have enough information in the documents."

Context:
{context}

Question:
{question}

Answer:
""")

# -----------------------------------
# Ask Question
# -----------------------------------
def ask_question(question):

    retrieved = retrieve(question)

    if not retrieved:
        return

    context = ""

    print("\nSource Documents:")
    print("-" * 50)

    for doc, meta in retrieved:

        print(meta["source"])

        context += (
            f"\nSource: {meta['source']}\n"
            f"{doc}\n"
        )

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAnswer:")
    print("-" * 50)
    print(answer)

# -----------------------------------
# Main Menu
# -----------------------------------
def main():

    while True:

        print("\nMini Knowledge Base Assistant")
        print("1. Upload Document")
        print("2. Ask Question")
        print("3. Exit")

        choice = input("\nChoose: ")

        if choice == "1":

            path = input(
                "Document path: "
            )

            try:
                index_document(path)

            except Exception as e:
                print("Error:", e)

        elif choice == "2":

            question = input(
                "Question: "
            )

            ask_question(question)

        elif choice == "3":
            break

        else:
            print("Invalid option")

# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":
    main()