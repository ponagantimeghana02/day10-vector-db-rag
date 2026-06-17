import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)

chunk_sizes = [200, 500, 1000]
top_k_values = [1, 3, 5]

embedding_models = [
    "all-MiniLM-L6-v2"
]

test_questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "How do vector databases work?"
]

prompt = ChatPromptTemplate.from_template("""
Use ONLY the context below.

Context:
{context}

Question:
{question}

If answer not found, say:
"I don't have enough information."

Answer:
""")

def evaluate_configuration(model_name, top_k):

    embedding_model = SentenceTransformer(model_name)

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("knowledge_base")

    chain = prompt | llm | StrOutputParser()

    results = []

    for question in test_questions:

        query_embedding = embedding_model.encode(question).tolist()

        retrieved = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        docs = retrieved["documents"][0]

        context = "\n".join(docs)

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        results.append({
            "question": question,
            "answer": answer
        })

    return results

report_lines = []

report_lines.append("# RAG Evaluation Report\n")

for emb_model in embedding_models:

    report_lines.append(f"\n## Embedding Model: {emb_model}\n")

    for top_k in top_k_values:

        report_lines.append(f"\n### Top K = {top_k}\n")

        outputs = evaluate_configuration(
            emb_model,
            top_k
        )

        for item in outputs:

            report_lines.append(
                f"Question: {item['question']}\n"
            )

            report_lines.append(
                f"Answer: {item['answer']}\n"
            )

            report_lines.append("---\n")

with open("evaluation_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("Evaluation complete.")
print("Generated: evaluation_report.md")