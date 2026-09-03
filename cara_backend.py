from sentence_transformers import SentenceTransformer, util
import pandas as pd


# Load routing history
comparison = pd.read_csv(
    "data/evaluated_routing_data.csv"
)

queries = comparison["query"].tolist()


# Load embedding model
embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# Create embeddings for historical queries
query_embeddings = embedding_model.encode(
    queries,
    convert_to_tensor=True
)


def find_similar_queries(new_query, top_k=3):

    new_embedding = embedding_model.encode(
        new_query,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        new_embedding,
        query_embeddings
    )[0]

    top_results = similarities.topk(
        k=min(top_k, len(queries))
    )

    similar_indices = top_results.indices.cpu().tolist()
    similarity_scores = top_results.values.cpu().tolist()

    results = []

    for index, score in zip(
        similar_indices,
        similarity_scores
    ):

        results.append({
            "query": comparison.iloc[index]["query"],
            "similarity": round(float(score), 3),
            "small_sufficient": int(
                comparison.iloc[index]["small_sufficient"]
            )
        })

    return results


def calculate_local_reliability(
    new_query,
    top_k=3
):

    similar_queries = find_similar_queries(
        new_query,
        top_k=top_k
    )

    weighted_success = 0
    total_weight = 0

    for item in similar_queries:

        similarity = item["similarity"]
        success = item["small_sufficient"]

        weighted_success += similarity * success
        total_weight += similarity

    if total_weight == 0:
        reliability = 0.5
    else:
        reliability = weighted_success / total_weight

    return reliability, similar_queries


def cara_route(
    query,
    threshold=0.70,
    top_k=3
):

    local_reliability, evidence = (
        calculate_local_reliability(
            query,
            top_k=top_k
        )
    )

    similarities = [
        item["similarity"]
        for item in evidence
    ]

    evidence_strength = (
        sum(similarities) /
        len(similarities)
    )

    cara_confidence = (
        local_reliability *
        evidence_strength
    )

    if cara_confidence >= threshold:
        route = "SMALL"
    else:
        route = "MEDIUM"

    return {
        "query": query,
        "route": route,
        "local_reliability": local_reliability,
        "evidence_strength": evidence_strength,
        "cara_confidence": cara_confidence,
        "evidence": evidence
    }