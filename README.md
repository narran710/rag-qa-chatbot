# 🤖 RAG QA Chatbot (Document Question Answering)

An advanced **Retrieval-Augmented Generation (RAG)** chatbot that answers questions from uploaded documents using semantic search + LLM reasoning.

---

## 🚀 Features

- 📄 Upload PDF / TXT documents
- 🔍 Hybrid Search (FAISS + BM25)
- 🤖 LLM-powered answers (Groq - Llama 3.1)
- 💬 Chat interface
- 📜 Chat history with:
  - Search
  - Delete selected / all
- ⚡ Fast retrieval using vector embeddings
- 🧠 Context-aware responses
- 📂 Source-based answering (answers strictly from document)

---

## 🧠 How It Works

1. Document is split into chunks
2. Converted into embeddings using Sentence Transformers
3. Stored in FAISS (vector DB)
4. BM25 used for keyword matching
5. Hybrid retrieval fetches best chunks
6. LLM generates answer from retrieved context

---

## 🛠 Tech Stack

- Python
- Streamlit
- FAISS
- Sentence Transformers
- BM25 (rank-bm25)
- Groq API (Llama 3.1)

---