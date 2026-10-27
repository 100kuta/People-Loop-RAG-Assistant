# PeopleLoop Remote Work Enablement Assistant

## Overview

This project implements an AI-powered employee support assistant using a **Retrieval-Augmented Generation (RAG)** architecture. The assistant answers employee questions **strictly based on the Remote Employee Handbook** and avoids hallucinations by grounding every response in the provided document.

The solution is designed for remote-first organizations and is built entirely with **free and open-source tools**.

---

## Key Features

* PDF ingestion and semantic search
* Document chunking with context preservation
* Vector embeddings and FAISS similarity search
* Document-grounded response generation
* Professionally friendly, enterprise-safe tone
* Chat-style user interface with Gradio
* Automatic input clearing and session reset
* Robust error handling

---

## Architecture Summary

1. **Document Ingestion** – Load and parse the Remote Employee Handbook (PDF)
2. **Chunking** – Split text into overlapping semantic chunks
3. **Embedding** – Convert chunks into vector embeddings
4. **Vector Store** – Store embeddings using FAISS
5. **Retrieval** – Retrieve relevant chunks per user query
6. **Generation** – Generate answers strictly from retrieved content

---

## Technology Stack

* **LLM**: google/flan-t5-base (open-source)
* **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
* **Vector Store**: FAISS
* **Frameworks**: LangChain (community modules), Transformers
* **UI**: Gradio (Blocks + Chatbot)
* **Deployment**: Hugging Face Spaces

---

## Project Structure

```
peopleloop-rag-assistant/
├── app.py
├── Employee_HB_Final.pdf
├── requirements.txt
├── README.md
└── vectorstore/
```

---

## Running Locally

1. Clone the repository

```bash
git clone https://github.com/100kuta/People-Loop-RAG-Assistant
cd peopleloop-rag-assistant
```

2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the application

```bash
python app.py
```

The app will open in your browser with a chat interface.

---

## Usage Notes

* The assistant answers questions **only** from the handbook
* If information is not present, it responds with:

  > "I don’t know based on the provided document."
* Models and embeddings are downloaded once and cached locally

---

## Deployment

This project is designed for easy deployment on **Hugging Face Spaces**:

* Space type: **Gradio**
* No API keys required
* Models are cached automatically by the platform

---

## License

This project is provided for assessment and demonstration purposes.
