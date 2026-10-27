# 🏢 PeopleLoop Remote Work Enablement Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/Orchestrator-LangChain-green.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/Vector_Store-FAISS-purple.svg)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/Model-google%2Fflan--t5--base-yellow.svg)](https://huggingface.co/google/flan-t5-base)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> A production-grade **Retrieval-Augmented Generation (RAG)** workplace companion designed for distributed organizations. The assistant grounds answers **strictly within the official Remote Employee Handbook** (`Employee_HB_Final.pdf`), delivering clear, factual support while enforcing negative constraints to prevent hallucinations. Built entirely using open-source, local-first models.

---

## 📌 Problem & Value Proposition

Distributed teams often struggle to navigate dense internal HR documentation, benefit policies, and remote-work guidelines. Traditional search returns disconnected paragraphs without synthesis, while general-purpose commercial LLMs risk hallucinating non-existent company benefits or leaking proprietary HR policies to third-party APIs.

**PeopleLoop Assistant** solves this through:
- **Strict Context Grounding:** The model only generates responses supported by the retrieved document chunks.
- **Deterministic Refusal:** If an employee asks about policies not documented in the handbook, the assistant explicitly states: *"I don’t know based on the provided document."*
- **100% Free & Open-Source:** Runs locally or on Hugging Face Spaces with zero external API fees.

---

## 🛠️ Architecture & Pipeline

```mermaid
graph TD
    A[Employee_HB_Final.pdf] --> B[PyPDFLoader]
    B --> C[RecursiveCharacterTextSplitter: 800 chars / 150 overlap]
    C --> D[Embeddings: all-MiniLM-L6-v2]
    D --> E[(FAISS Vector Index)]
    
    Q[Employee Query] --> RET[Similarity Search: Top-K=2]
    E --> RET
    RET --> CHK{Passages Found?}
    CHK -->|No| REF[Refusal: I don't know based on provided document]
    CHK -->|Yes| CTX[Structured Context Prompt]
    CTX --> LLM[HuggingFace Transformers: google/flan-t5-base]
    LLM --> UI[Gradio Chat Interface]
```

### Technical Workflow:
1. **Document Ingestion & Chunking:** Loads and parses `Employee_HB_Final.pdf` using LangChain's `PyPDFLoader`. Text is split into overlapping chunks (size: 800 tokens, overlap: 150 tokens) to preserve semantic continuity across section boundaries.
2. **Dense Vector Embeddings:** Chunks are vectorized using `sentence-transformers/all-MiniLM-L6-v2`, yielding high retrieval accuracy with low computational overhead.
3. **FAISS Similarity Search:** Fast in-memory local vector store (`vectorstore/faiss_index`) with local caching to eliminate re-indexing latency on repeated runs.
4. **Context-Constrained Prompting:** Formats retrieved passages into a structured system prompt directing `google/flan-t5-base` to summarize strictly without speculation.
5. **Interactive UI:** Served via a responsive, mobile-friendly **Gradio Chatbot** with automatic input clearing and session management.

---

## 📂 Project Structure

```text
People-Loop-RAG-Assistant/
├── app.py                   # Core RAG pipeline, LLM inference, and Gradio chat interface
├── Employee_HB_Final.pdf    # Source corporate remote work handbook
├── requirements.txt         # Production dependencies
├── .gitignore               # Ignored cache and vector index files
└── README.md                # Technical documentation
```

---

## 🚀 Quickstart & Local Setup

### 1. Prerequisites
- Python 3.10 or 3.11
- Git

### 2. Installation
```bash
git clone https://github.com/100kuta/People-Loop-RAG-Assistant.git
cd People-Loop-RAG-Assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Assistant
```bash
python app.py
```
The Gradio web interface will launch automatically at `http://127.0.0.1:7860`.

---

## 💬 Sample Evaluation Queries

| Employee Query | Grounded System Behavior |
|---|---|
| *"What is our policy on core working hours?"* | Retrieves remote working hour clause; summarizes expectations clearly. |
| *"How do I claim home office equipment reimbursement?"* | Extracts expense policy and step-by-step submission steps. |
| *"What is the annual paternity leave allowance?"* | Answers strictly if stated in the handbook. |
| *"What is the company stock option vesting schedule?"* | Emits refusal: *"I don’t know based on the provided document."* (Prevents speculation). |

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
