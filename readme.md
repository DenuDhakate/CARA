# 🧠 CARA

## Local Calibration-Aware Adaptive Routing for Multi-LLM Systems

CARA is an adaptive Large Language Model (LLM) routing system that dynamically selects between smaller and larger language models based on the characteristics of an incoming query.

Instead of sending every query to a computationally expensive model, CARA analyzes semantic similarity with historical queries, estimates local model reliability, evaluates available evidence, and produces a routing confidence score.

The system can route simpler or familiar queries to a smaller model while escalating uncertain queries to a more capable model.

---

## 🚀 Key Idea

Traditional multi-LLM systems can use fixed routing strategies.

For example:

    Every Query
         ↓
    Large Model

This can unnecessarily increase computational usage.

CARA instead follows:

    User Query
         │
         ▼
    Semantic Analysis
         │
         ▼
    Historical Evidence
         │
         ▼
    Local Reliability
         │
         ▼
    Evidence Strength
         │
         ▼
    CARA Confidence
         │
      ┌──┴──┐
      │     │
     High   Low
      │     │
      ▼     ▼
    SMALL  MEDIUM
    MODEL  MODEL

---

## 🎯 Objectives

The main objectives of CARA are:

- Analyze incoming queries semantically.
- Retrieve similar historical queries.
- Estimate local reliability of the smaller model.
- Measure the strength of available evidence.
- Calculate an adaptive routing confidence.
- Select an appropriate language model.
- Reduce unnecessary use of larger models.
- Provide an interpretable explanation for routing decisions.

---

## 🧠 CARA Architecture

CARA consists of several major components.

![CARA Architecture](docs/cara_architecture.png)

### 1. Query Input

The user submits a natural-language query through the Streamlit interface.

### 2. Semantic Representation

CARA converts the query into an embedding using:

    all-MiniLM-L6-v2

### 3. Historical Evidence Retrieval

The new query is compared with previously evaluated queries using cosine similarity.

The most similar historical queries are selected as local evidence.

### 4. Local Reliability

CARA examines whether the smaller model was sufficient for similar historical queries.

A weighted reliability score is calculated using semantic similarity.

### 5. Evidence Strength

The average semantic similarity of retrieved historical examples is used as an indication of how strongly the current query is represented by historical evidence.

### 6. Routing Confidence

CARA combines local reliability and evidence strength into an adaptive confidence score.

The current implementation uses:

    CARA Confidence =
    0.7 × Local Reliability
    +
    0.3 × Evidence Strength

### 7. Adaptive Routing

Based on the routing signals, CARA selects between:

- Qwen2.5-0.5B-Instruct
- Qwen2.5-1.5B-Instruct

### 8. Response Generation

The selected model generates the final response.

---

## 🔬 Research-Oriented Features

### Local Reliability

Rather than relying only on a global model capability assumption, CARA examines historically similar queries.

### Semantic Evidence

Historical examples are retrieved according to semantic similarity rather than simple keyword matching.

### Confidence-Aware Routing

The system produces an explicit confidence score for its routing decision.

### Evidence-Aware Escalation

When available historical evidence is weak, CARA can escalate the query to the larger model.

### Familiarity Signal

The system evaluates how familiar the incoming query is relative to historical routing data.

---

## 🤖 Models

### Embedding Model

    all-MiniLM-L6-v2

Used for semantic representation and similarity search.

### Small Language Model

    Qwen/Qwen2.5-0.5B-Instruct

### Medium Language Model

    Qwen/Qwen2.5-1.5B-Instruct

---

## 📊 Evaluation

CARA was evaluated using a test set of 10 queries.

The current evaluation produced:

| Metric | Result |
|---|---:|
| Total Queries | 10 |
| Routing Accuracy | 90% |
| Small Model Usage | 50% |
| Medium Model Usage | 50% |
| Average Confidence | 0.600 |
| Average Local Reliability | 0.618 |
| Average Evidence Strength | 0.574 |
| Average Familiarity | 0.829 |

> These results represent the current 10-query evaluation and should not be interpreted as evidence of general performance across all possible queries.

---

## 📈 Evaluation Visualizations

The evaluation pipeline generates the following figures:

    evaluation/
    └── figures/
        ├── routing_accuracy.png
        ├── model_usage.png
        └── cara_metrics.png

These visualizations summarize routing accuracy, model-selection distribution, and CARA's confidence-related metrics.


### Routing Accuracy

![CARA Routing Accuracy](evaluation/figures/routing_accuracy.png)

### Model Routing Distribution

![CARA Model Usage](evaluation/figures/model_usage.png)

### CARA Intelligence Metrics

![CARA Metrics](evaluation/figures/cara_metrics.png)

---

## 🖥️ Technology Stack

### Programming

- Python

### Machine Learning

- PyTorch
- Hugging Face Transformers
- Sentence Transformers

### Data Processing

- Pandas

### Interface

- Streamlit

### Models

- Qwen2.5
- all-MiniLM-L6-v2

---

## 📁 Project Structure

    CARA/
    │
    ├── app.py
    ├── cara_backend.py
    ├── model_manager.py
    ├── requirements.txt
    ├── README.md
    │
    ├── data/
    │   └── evaluated_routing_data.csv
    │
    └── evaluation/
        │
        ├── evaluate_cara.py
        ├── visualize_results.py
        ├── cara_evaluation_results.csv
        │
        └── figures/
            ├── routing_accuracy.png
            ├── model_usage.png
            └── cara_metrics.png

---

## ⚙️ Installation

### 1. Clone the repository

    git clone https://github.com/DenuDhakate/CARA.git

    cd CARA

### 2. Create the environment

    conda create -n cara python=3.11

Activate it:

    conda activate cara

### 3. Install dependencies

    pip install -r requirements.txt

---

## ▶️ Running CARA

Launch the Streamlit application:

    streamlit run app.py

The application will open in your browser.

---

## 🧪 Running the Evaluation

Run:

    python evaluation/evaluate_cara.py

The evaluation results are saved to:

    evaluation/cara_evaluation_results.csv

Generate the evaluation graphs:

    python evaluation/visualize_results.py

The figures are saved to:

    evaluation/figures/

---

## 🔍 Example

### Simple Query

Input:

    2 + 2

CARA identifies the request as computationally simple and can route it to the smaller model.

### More Complex Query

Input:

    Explain how a neural network learns using backpropagation.

CARA evaluates historical evidence and routing confidence before selecting the appropriate model.

---

## 🛡️ Why Adaptive Routing?

Large language models generally require more computational resources than smaller models.

If a smaller model can successfully handle a query, routing that query to the smaller model can potentially reduce unnecessary computational usage.

CARA therefore investigates the following principle:

    Use the smallest capable model
    when confidence and evidence support the decision.

When evidence is insufficient or confidence is low, the system can escalate the query.

---

## ⚠️ Current Limitations

The current implementation has several limitations:

- The evaluation dataset is relatively small.
- Routing accuracy depends on the quality of historical routing data.
- The embedding model is fixed.
- Only two language-model sizes are currently used.
- The current evaluation does not represent all possible query distributions.
- Further experiments are required to evaluate performance under distribution shift.
- Computational cost and latency measurements can be expanded in future work.

---

## 🔮 Future Work

Possible future extensions include:

- Larger evaluation datasets.
- More diverse language models.
- Dynamic confidence calibration.
- Explicit distribution-shift detection.
- Budget-aware routing.
- Selective verification of uncertain responses.
- Multi-model routing with more than two model sizes.
- Latency and computational-cost analysis.
- Online learning from newly evaluated queries.

---

## 👥 Project

CARA is developed as an academic/research-oriented project exploring adaptive routing for multi-LLM systems.

The project focuses on combining semantic evidence, local reliability, and confidence-aware decision making to create an interpretable adaptive routing mechanism.

---

## 📜 License

This project is intended for academic and research purposes.