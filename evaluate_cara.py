import sys
import os
import pandas as pd

# Allow Python to find cara_backend.py
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

from cara_backend import cara_route


# =========================================================
# LOAD DATASET
# =========================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "evaluated_routing_data.csv"
)

df = pd.read_csv(DATA_PATH)


# =========================================================
# RUN CARA
# =========================================================

results = []

print("\n==========================================")
print("        CARA EVALUATION STARTED")
print("==========================================\n")


for index, row in df.iterrows():

    query = str(row["query"])

    print(
        f"[{index + 1}/{len(df)}] "
        f"Analyzing: {query}"
    )

    try:

        decision = cara_route(query)

        results.append({

            "query": query,

            "actual_small_sufficient":
                int(row["small_sufficient"]),

            "route":
                decision["route"],

            "local_reliability":
                decision["local_reliability"],

            "evidence_strength":
                decision["evidence_strength"],

            "familiarity":
                decision["familiarity"],

            "shift_penalty":
                decision["shift_penalty"],

            "cara_confidence":
                decision["cara_confidence"],

            "confidence_status":
                decision["confidence_status"]
        })

    except Exception as e:

        print(
            f"ERROR processing query: {e}"
        )


# =========================================================
# CREATE RESULTS DATAFRAME
# =========================================================

results_df = pd.DataFrame(results)


# =========================================================
# ROUTING CORRECTNESS
# =========================================================

results_df["predicted_small"] = (
    results_df["route"] == "SMALL"
).astype(int)


results_df["routing_correct"] = (
    results_df["predicted_small"]
    ==
    results_df["actual_small_sufficient"]
)


routing_accuracy = (
    results_df["routing_correct"].mean()
)


# =========================================================
# MODEL USAGE
# =========================================================

small_usage = (
    results_df["route"]
    .eq("SMALL")
    .mean()
)

medium_usage = (
    results_df["route"]
    .eq("MEDIUM")
    .mean()
)


# =========================================================
# AVERAGE METRICS
# =========================================================

average_confidence = (
    results_df["cara_confidence"]
    .mean()
)

average_reliability = (
    results_df["local_reliability"]
    .mean()
)

average_evidence = (
    results_df["evidence_strength"]
    .mean()
)

average_familiarity = (
    results_df["familiarity"]
    .mean()
)

average_shift_penalty = (
    results_df["shift_penalty"]
    .mean()
)


# =========================================================
# SAVE RESULTS
# =========================================================

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "cara_evaluation_results.csv"
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n")
print("==========================================")
print("          CARA EVALUATION RESULTS")
print("==========================================")

print(
    f"\nTotal Queries        : "
    f"{len(results_df)}"
)

print(
    f"Routing Accuracy     : "
    f"{routing_accuracy * 100:.2f}%"
)

print(
    f"SMALL Model Usage    : "
    f"{small_usage * 100:.2f}%"
)

print(
    f"MEDIUM Model Usage   : "
    f"{medium_usage * 100:.2f}%"
)

print(
    f"\nAverage Confidence   : "
    f"{average_confidence:.3f}"
)

print(
    f"Average Reliability  : "
    f"{average_reliability:.3f}"
)

print(
    f"Average Evidence    : "
    f"{average_evidence:.3f}"
)

print(
    f"Average Familiarity : "
    f"{average_familiarity:.3f}"
)

print(
    f"Average Shift Penalty:"
    f" {average_shift_penalty:.3f}"
)

print("\n==========================================")

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_PATH}"
)

print("\n==========================================\n")