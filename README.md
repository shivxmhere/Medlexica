# MedLexica

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-EE4C2C.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![W&B](https://img.shields.io/badge/Weights_&_Biases-FFBE00?logo=weightsandbiases&logoColor=white)

## Overview
MedLexica is a production-grade clinical AI system. It utilizes semantic vector search, knowledge graphs, and cross-encoder re-ranking in a hybrid RAG pipeline. Powered by a QLoRA fine-tuned Phi-3 model, it serves highly accurate, locally hosted inference with citation capabilities and hallucination guardrails via FastAPI.

## Architecture Diagram

```text
[Clinical Query]
     │
[Query Pre-processor] ── tokenize · de-identify · intent classify
     │
┌────┴────────────────┐─────────────────────┐
│                     │                     │
[Vector Search]  [Knowledge Graph]  [Structured DB]
FAISS + BM25     ICD-10 ontology    Lab refs · dosage
│                     │                     │
└────────────┬─────────────────────┘
             │
[Cross-Encoder Re-Ranker]
ms-marco MiniLM · top-K selection
             │
[MedLexica Core - Phi-3-mini-4k]
QLoRA fine-tuned · 4-bit quantized
             │
[Output + Guardrails]
Hallucination score · citation · JSON
             │
     [FastAPI · Docker]
```

## Setup Instructions
1. Clone the repository.
   ```bash
   git clone https://github.com/shivxmhere/Medlexica.git
   cd Medlexica
   ```
2. Create and activate a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Configure environment variables.
   ```bash
   cp .env.example .env
   # Update variables inside .env appropriately
   ```
5. Run the API.
   ```bash
   uvicorn api.main:app --reload
   ```

## Results

| Metric | Base Phi-3 | MedLexica | Delta |
|--------|-----------|-----------|-------|
| ROUGE-L | TBD | TBD | TBD |
| BERTScore F1 | TBD | TBD | TBD |
| Hallucination Rate | TBD | TBD | TBD |
| ICD-10 Accuracy | TBD | TBD | TBD |
| P95 Latency (ms) | TBD | TBD | TBD |

## Technical Deep-Dive
- **QLoRA Fine-Tuning:** The Phi-3-mini-4k model is fine-tuned using Quantized Low-Rank Adaptation (QLoRA) utilizing 4-bit quantization, balancing performance with resource efficiency.
- **Hybrid RAG Pipeline:** Combines dense FAISS retrievers and sparse BM25 indices to fetch the most relevant knowledge alongside structured databases and ICD-10 ontology.
- **Cross-Encoder Reranking:** Retrieved documents are prioritized utilizing an ms-marco MiniLM cross-encoder to refine context mapping before the generative phase.

## Reproducibility
### Training Steps
1. Populate `data/raw/` with designated datasets.
2. Build datasets using `python training/dataset_builder.py`.
3. Start training via `python training/train.py`.
4. Monitor progress within the [W&B Dashboard](https://wandb.ai/). (Placeholder Link)
