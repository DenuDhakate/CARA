# 🧠 CARA

## Local Calibration-Aware Adaptive Routing for Multi-LLM Systems

CARA is an experimental adaptive routing system designed to select an appropriate Large Language Model (LLM) for a given user query.

Instead of sending every query to the same model, CARA analyzes historical evidence from semantically similar queries and estimates whether a smaller model is likely to be sufficient.

Based on this analysis, the system adaptively routes the query to either a Small or Medium language model.

---

## 🚀 Features

- Adaptive routing between multiple LLMs
- Small Model: Qwen2.5-0.5B-Instruct
- Medium Model: Qwen2.5-1.5B-Instruct
- Semantic similarity-based query retrieval
- Historical routing evidence
- Local reliability estimation
- Evidence strength estimation
- Adaptive SMALL or MEDIUM model selection
- Automatic model switching
- GPU-based local inference
- Interactive Streamlit user interface

---

## 🏗️ System Architecture

```text
User Query
    │
    ▼
Semantic Similarity Retrieval
    │
    ▼
Historical Query Evidence
    │
    ▼
Local Reliability Estimation
    │
    ▼
Evidence Strength
    │
    ▼
CARA Confidence
    │
    ▼
Adaptive Routing
    │
    ├───────────────┐
    ▼               ▼
SMALL LLM       MEDIUM LLM
Qwen 0.5B       Qwen 1.5B
    │               │
    └───────┬───────┘
            ▼
    Generated Response