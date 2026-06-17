import os
import chromadb
from sentence_transformers import SentenceTransformer

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=os.getenv("GROQ_API_KEY")
    # temperature=0
)


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="knowledge_base")

def retrieve_docs(query, k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]

    return list(zip(docs, metas))

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""")

def rag_pipeline(question):
    retrieved_docs = retrieve_docs(question, k=3)

    context_text = ""

    for doc, meta in retrieved_docs:
        context_text += f"\nSource: {meta.get('source', 'Unknown')}\nText: {doc}\n"

    # If no useful context
    if len(context_text.strip()) == 0:
        return "I don't have enough information in the provided documents."

    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "context": context_text,
        "question": question
    })

    return response


print("🔥 Basic RAG Pipeline Ready!")
print("-" * 50)

while True:
    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    answer = rag_pipeline(query)

    print("\n🧠 Answer:")
    print("-" * 50)
    print(answer)