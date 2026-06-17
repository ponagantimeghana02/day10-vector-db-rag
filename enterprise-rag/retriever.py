# retriever.py

from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)

# from rank_bm25 import BM25Okapi

from vector_store import (
    get_collection
)

from config import (
    EMBEDDING_MODEL,
    RERANK_MODEL,
    TOP_K,
    RERANK_TOP_K
)


# ==================================
# Models
# ==================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

reranker = CrossEncoder(
    RERANK_MODEL
)

collection = get_collection()


# ==================================
# Vector Search
# ==================================

def vector_search(
    query,
    top_k=TOP_K,
    metadata_filter=None
):

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    if metadata_filter:

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k,
            where=metadata_filter
        )

    else:

        results = collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=top_k
        )

    return results


# ==================================
# BM25 Search
# ==================================

# def bm25_search(
#     query,
#     top_k=TOP_K
# ):

#     data = collection.get()

#     docs = data["documents"]

#     tokenized_docs = [
#         doc.split()
#         for doc in docs
#     ]

#     bm25 = BM25Okapi(
#         tokenized_docs
#     )

#     scores = bm25.get_scores(
#         query.split()
#     )

#     ranked = sorted(
#         zip(docs, scores),
#         key=lambda x: x[1],
#         reverse=True
#     )

#     return ranked[:top_k]


# ==================================
# Hybrid Search
# ==================================

# def hybrid_search(
#     query,
#     top_k=TOP_K
# ):

    # vector_results = vector_search(
    #     query,
    #     top_k
    # )

    # docs = vector_results[
    #     "documents"
    # ][0]

    # metas = vector_results[
    #     "metadatas"
    # ][0]

    # bm25_results = bm25_search(
    #     query,
    #     top_k
    # )

    # hybrid_docs = []

    # for doc, meta in zip(
    #     docs,
    #     metas
    # ):

    #     hybrid_docs.append({
    #         "document": doc,
    #         "metadata": meta
    #     })

    # for doc, score in bm25_results:

    #     hybrid_docs.append({
    #         "document": doc,
    #         "metadata": {
    #             "bm25_score": score
    #         }
    #     })

    # return hybrid_docs


# ==================================
# Re-Ranking
# ==================================

def rerank_results(
    query,
    retrieved_docs
):

    pairs = []

    for item in retrieved_docs:

        pairs.append(
            (
                query,
                item["document"]
            )
        )

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(
            retrieved_docs,
            scores
        ),
        key=lambda x: x[1],
        reverse=True
    )

    final_results = []

    for item, score in ranked[
        :RERANK_TOP_K
    ]:

        item["rerank_score"] = (
            float(score)
        )

        final_results.append(
            item
        )

    return final_results


# ==================================
# Full Retrieval Pipeline
# ==================================

def retrieve_context(
    query,
    metadata_filter=None
):

    vector_results = vector_search(
        query=query,
        top_k=TOP_K,
        metadata_filter=metadata_filter
    )

    docs = vector_results[
        "documents"
    ][0]

    metas = vector_results[
        "metadatas"
    ][0]

    retrieved_docs = []

    for doc, meta in zip(
        docs,
        metas
    ):

        retrieved_docs.append({

            "document": doc,

            "metadata": meta

        })

    reranked = rerank_results(

        query,

        retrieved_docs

    )

    return reranked


# ==================================
# Build Context
# ==================================

def build_context(
    retrieved_docs
):

    context = ""

    sources = []

    for item in retrieved_docs:

        context += (
            item["document"]
            + "\n\n"
        )

        source = item[
            "metadata"
        ].get(
            "source",
            "Unknown"
        )

        sources.append(
            source
        )

    return context, sources


# ==================================
# Test
# ==================================

if __name__ == "__main__":

    query = input(
        "Question: "
    )

    results = retrieve_context(
        query
    )

    print("\nResults")

    print("-" * 50)

    for item in results:

        print(
            item["document"][:300]
        )

        print()

        print(
            item["metadata"]
        )

        print("-" * 50)