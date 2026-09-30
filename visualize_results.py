import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# LOAD RESULTS
# =========================================================

BASE_DIR = os.path.dirname(__file__)

RESULTS_FILE = os.path.join(
    BASE_DIR,
    "cara_evaluation_results.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


df = pd.read_csv(RESULTS_FILE)


# =========================================================
# CALCULATE SUMMARY
# =========================================================

routing_accuracy = (
    df["routing_correct"].mean() * 100
)

small_usage = (
    df["route"].eq("SMALL").mean() * 100
)

medium_usage = (
    df["route"].eq("MEDIUM").mean() * 100
)


# =========================================================
# GRAPH 1 — ROUTING ACCURACY
# =========================================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["Correct Routing", "Incorrect Routing"],
    [
        routing_accuracy,
        100 - routing_accuracy
    ]
)

plt.ylabel("Percentage (%)")
plt.title("CARA Routing Accuracy")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "routing_accuracy.png"
    ),
    dpi=200
)

plt.close()


# =========================================================
# GRAPH 2 — MODEL USAGE
# =========================================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["Small Model", "Medium Model"],
    [
        small_usage,
        medium_usage
    ]
)

plt.ylabel("Usage (%)")
plt.title("CARA Model Routing Distribution")
plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_usage.png"
    ),
    dpi=200
)

plt.close()


# =========================================================
# GRAPH 3 — CARA INTELLIGENCE METRICS
# =========================================================

metrics = [
    "cara_confidence",
    "local_reliability",
    "evidence_strength",
    "familiarity"
]

values = [
    df["cara_confidence"].mean(),
    df["local_reliability"].mean(),
    df["evidence_strength"].mean(),
    df["familiarity"].mean()
]

labels = [
    "Confidence",
    "Reliability",
    "Evidence",
    "Familiarity"
]


plt.figure(figsize=(8, 5))

plt.bar(
    labels,
    values
)

plt.ylabel("Score")
plt.title("CARA Intelligence Metrics")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "cara_metrics.png"
    ),
    dpi=200
)

plt.close()


# =========================================================
# PRINT SUMMARY
# =========================================================

print()
print("=" * 50)
print("        CARA VISUAL EVALUATION")
print("=" * 50)

print(
    f"\nRouting Accuracy : "
    f"{routing_accuracy:.2f}%"
)

print(
    f"Small Usage      : "
    f"{small_usage:.2f}%"
)

print(
    f"Medium Usage     : "
    f"{medium_usage:.2f}%"
)

print(
    f"\nAverage Confidence : "
    f"{df['cara_confidence'].mean():.3f}"
)

print(
    f"Average Reliability: "
    f"{df['local_reliability'].mean():.3f}"
)

print(
    f"Average Evidence   : "
    f"{df['evidence_strength'].mean():.3f}"
)

print(
    f"Average Familiarity: "
    f"{df['familiarity'].mean():.3f}"
)

print()
print("Graphs generated successfully!")
print()
print("Saved inside:")
print(OUTPUT_DIR)
print("=" * 50)