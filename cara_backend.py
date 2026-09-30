from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re


# =================================================
# LOAD HISTORICAL ROUTING DATA
# =================================================

comparison = pd.read_csv(
    "data/evaluated_routing_data.csv"
)

queries = comparison["query"].tolist()


# =================================================
# LOAD EMBEDDING MODEL
# =================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =================================================
# CREATE EMBEDDINGS FOR HISTORICAL QUERIES
# =================================================

query_embeddings = embedding_model.encode(
    queries,
    convert_to_tensor=True
)


# =================================================
# SIMPLE QUERY DETECTION
# =================================================

def is_simple_query(query):

    query = query.strip().lower()

    # Pure arithmetic expressions
    #
    # Examples:
    # 2 + 2
    # 5 * 10
    # 100 / 5
    # (20 + 5) * 2

    if re.fullmatch(
        r"[0-9\s\+\-\*\/\(\)\.\%]+",
        query
    ):
        return True

    # Very short queries are generally
    # computationally/simple informational
    # requests.

    if len(query.split()) <= 3:
        return True

    return False


# =================================================
# FIND SIMILAR HISTORICAL QUERIES
# =================================================

def find_similar_queries(
    new_query,
    top_k=3
):

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

    similar_indices = (
        top_results.indices.cpu().tolist()
    )

    similarity_scores = (
        top_results.values.cpu().tolist()
    )

    results = []

    for index, score in zip(
        similar_indices,
        similarity_scores
    ):

        results.append({

            "query":
                comparison.iloc[index]["query"],

            "similarity":
                round(
                    float(score),
                    3
                ),

            "small_sufficient":
                int(
                    comparison.iloc[index][
                        "small_sufficient"
                    ]
                )
        })

    return results


# =================================================
# CALCULATE LOCAL RELIABILITY
# =================================================

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

        success = item[
            "small_sufficient"
        ]

        # Only positive similarity should
        # contribute to the reliability estimate.

        weight = max(
            similarity,
            0
        )

        weighted_success += (
            weight * success
        )

        total_weight += weight

    if total_weight == 0:

        reliability = 0.5

    else:

        reliability = (
            weighted_success /
            total_weight
        )

    return (
        reliability,
        similar_queries
    )


# =================================================
# CALCULATE EVIDENCE STRENGTH
# =================================================

def calculate_evidence_strength(
    evidence
):

    if not evidence:

        return 0.0

    similarities = [
        item["similarity"]
        for item in evidence
    ]

    return (
        sum(similarities) /
        len(similarities)
    )


# =================================================
# DISTRIBUTION-SHIFT / FAMILIARITY SIGNAL
# =================================================

def calculate_familiarity(
    evidence
):

    """
    Measures how familiar the current query is
    compared with historical routing experience.

    Higher similarity =
    more familiar query region.

    Lower similarity =
    more unfamiliar / shifted query region.
    """

    if not evidence:

        return 0.0

    similarities = [
        item["similarity"]
        for item in evidence
    ]

    # Strongest semantic match

    max_similarity = max(
        similarities
    )

    # Average semantic similarity

    average_similarity = (
        sum(similarities) /
        len(similarities)
    )

    # Combine strongest evidence and
    # overall neighborhood similarity.

    familiarity = (
        0.6 * max_similarity
        +
        0.4 * average_similarity
    )

    # Normalize to [0, 1]

    familiarity = max(
        0.0,
        min(
            1.0,
            familiarity
        )
    )

    return familiarity


# =================================================
# DISTRIBUTION-SHIFT PENALTY
# =================================================

def calculate_shift_penalty(
    familiarity,
    shift_threshold=0.40
):

    """
    When familiarity is low, CARA becomes
    less confident in its historical evidence.

    This prevents high local reliability from
    automatically producing high confidence when
    the query is outside the historical region.
    """

    if familiarity >= shift_threshold:

        penalty = 0.0

    else:

        # Maximum penalty for very unfamiliar
        # queries.

        penalty = (
            shift_threshold
            - familiarity
        ) / shift_threshold

    return max(
        0.0,
        min(
            1.0,
            penalty
        )
    )


# =================================================
# CARA ADAPTIVE ROUTING
# =================================================

def cara_route(
    query,
    threshold=0.70,
    top_k=3,
    min_evidence=0.40,
    shift_threshold=0.40
):

    # =================================================
    # STEP 1 — SIMPLE QUERY DETECTION
    # =================================================

    simple_query = is_simple_query(
        query
    )


    # =================================================
    # STEP 2 — LOCAL HISTORICAL EVIDENCE
    # =================================================

    local_reliability, evidence = (
        calculate_local_reliability(
            query,
            top_k=top_k
        )
    )


    # =================================================
    # STEP 3 — EVIDENCE STRENGTH
    # =================================================

    evidence_strength = (
        calculate_evidence_strength(
            evidence
        )
    )


    # =================================================
    # STEP 4 — QUERY FAMILIARITY
    # =================================================

    familiarity = (
        calculate_familiarity(
            evidence
        )
    )


    # =================================================
    # STEP 5 — DISTRIBUTION SHIFT
    # =================================================

    shift_penalty = (
        calculate_shift_penalty(
            familiarity,
            shift_threshold
        )
    )


    # =================================================
    # STEP 6 — BASE CONFIDENCE
    # =================================================

    base_confidence = (

        0.60 *
        local_reliability

        +

        0.40 *
        evidence_strength
    )


    # =================================================
    # STEP 7 — SHIFT-AWARE CONFIDENCE
    # =================================================

    cara_confidence = (
        base_confidence
        *
        (1.0 - shift_penalty)
    )


    # =================================================
    # STEP 8 — KEEP CONFIDENCE BOUNDED
    # =================================================

    cara_confidence = max(
        0.0,
        min(
            1.0,
            cara_confidence
        )
    )


    # =================================================
    # STEP 9 — FINAL ROUTING DECISION
    # =================================================

    if simple_query:

        route = "SMALL"

        confidence_status = (
            "SIMPLE QUERY"
        )

        decision_reason = (
            "CARA detected a computationally simple "
            "or very short query. The Small model is "
            "sufficient for this request."
        )


    elif evidence_strength < min_evidence:

        route = "MEDIUM"

        confidence_status = (
            "LOW EVIDENCE"
        )

        decision_reason = (
            "CARA found insufficiently similar "
            "historical routing evidence. The query "
            "is treated as an unfamiliar semantic "
            "region and is conservatively routed "
            "to the Medium model."
        )


    elif shift_penalty > 0.0:

        route = "MEDIUM"

        confidence_status = (
            "DISTRIBUTION SHIFT"
        )

        decision_reason = (
            "The query shows limited familiarity "
            "with CARA's historical routing experience. "
            "CARA reduces confidence because the query "
            "may represent a distribution-shifted input "
            "and therefore routes it to the Medium model."
        )


    elif cara_confidence >= threshold:

        route = "SMALL"

        confidence_status = (
            "HIGH CONFIDENCE"
        )

        decision_reason = (
            "Historical routing evidence is sufficiently "
            "strong and locally reliable. CARA estimates "
            "that the Small model is sufficient."
        )


    else:

        route = "MEDIUM"

        confidence_status = (
            "ESCALATED"
        )

        decision_reason = (
            "Historical evidence was available, but "
            "shift-aware CARA confidence did not reach "
            "the routing threshold. The query was "
            "therefore escalated to the Medium model."
        )


    # =================================================
    # STEP 10 — RETURN COMPLETE DECISION
    # =================================================

    return {

        "query":
            query,

        "route":
            route,

        "local_reliability":
            round(
                float(
                    local_reliability
                ),
                3
            ),

        "evidence_strength":
            round(
                float(
                    evidence_strength
                ),
                3
            ),

        "familiarity":
            round(
                float(
                    familiarity
                ),
                3
            ),

        "shift_penalty":
            round(
                float(
                    shift_penalty
                ),
                3
            ),

        "cara_confidence":
            round(
                float(
                    cara_confidence
                ),
                3
            ),

        "confidence_status":
            confidence_status,

        "decision_reason":
            decision_reason,

        "evidence":
            evidence
    }