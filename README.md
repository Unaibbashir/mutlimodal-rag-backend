# Multimodal Clinical RAG Backend

An enterprise-grade, AI-powered dermatology consultation engine built with **FastAPI**, **LangGraph**, **PyTorch (EfficientNet-B0)**, **Pinecone**, and **Supabase (PostgreSQL + Storage)**. It analyzes patient skin imagery alongside clinical queries, retrieves evidence-based dermatological knowledge, and streams grounded responses in real time.

---

##  System Architecture

```text
┌───────────────────┐      Multipart/Form-Data      ┌──────────────────────────┐
│  Next.js Client   │ ────────────────────────────► │     FastAPI Gateway      │
│ (Auth & Chat UI)  │ ◄──────────────────────────── │   (/api/chat & /stream)  │
└───────────────────┘      Real-Time SSE Stream     └────────────┬─────────────┘
                                                                 │
                         ┌───────────────────────────────────────┼───────────────────────────────────────┐
                         ▼                                       ▼                                       ▼
          ┌─────────────────────────────┐         ┌─────────────────────────────┐         ┌─────────────────────────────┐
          │   Vision Inference Engine   │         │   Agentic RAG Workflow      │         │    Persistence & Storage    │
          │  PyTorch (EfficientNet-B0)  │         │  LangGraph + Pinecone + LLM │         │   Supabase Postgres/Bucket  │
          └─────────────────────────────┘         └─────────────────────────────┘         └─────────────────────────────┘
```

---

##  Key Features

- **Multimodal Vision + Text Pipeline:** Uses a fine-tuned `EfficientNet-B0` PyTorch classifier to detect potential skin conditions from uploaded images and enrich semantic retrieval queries.
- **Dual-Stage RAG Architecture (Local FAISS → Cloud Pinecone):** Benchmark and tune chunking/retrieval hyperparameters offline using FAISS before promoting optimal settings to Pinecone.
- **Persistent Multi-Thread Memory:** Powered by LangGraph's `PostgresSaver` backed by Supabase PostgreSQL, allowing patients to maintain isolated consultation threads and resume anytime.
- **Real-Time SSE Streaming:** Streams vision prediction badges and `gpt-4o-mini` tokens live via Server-Sent Events (`/api/chat/stream`).
- **Anti-Hallucination & Context Guardrails:** Enforces a strict **20-message (10-turn) cap** per consultation thread to prevent context drift and maintain clinical accuracy.
- **Right-to-Erasure Compliance:** `DELETE /api/threads/{user_id}/{thread_id}` permanently purges thread metadata, PostgreSQL checkpoints (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`), and patient skin images from Supabase Storage.

---

##  RAG Experimentation & Hyperparameter Tuning

Before indexing clinical documents into the cloud vector store, you can evaluate and tune retrieval parameters locally without cloud overhead:

1. **Local Evaluation & Tuning (`test_retriever.py` & `test_generator.py`):**
   - Uses a **local FAISS vector database** for rapid, offline experimentation.
   - Tune `chunk_size`, `chunk_overlap`, top-`k` retrieval count, embedding models, and clinical prompt templates.
   - Validate retrieval relevance and response faithfulness before deployment.

2. **Promoting Tuned Parameters to Production (`retriever.py` & `generator.py`):**
   - Transfer validated `chunk_size`, `chunk_overlap`, and `k` parameters into `retriever.py`.
   - Production consultations query **Pinecone** (serverless vector database) paired with `generator.py`.

---

##  Prerequisites & Environment Setup

### 1. Clone & Activate Virtual Environment
```bash
git clone https://github.com
cd mutlimodal-rag-backend

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory:
```env
# OpenAI & Pinecone
OPENAI_API_KEY="your-openai-api-key"
PINECONE_API_KEY="your-pinecone-api-key"

# Supabase Storage & Database
SUPABASE_URL="https://your-project-ref.supabase.co"
SUPABASE_KEY="your-supabase-service-role-or-secret-key"
SUPABASE_SERVICE_ROLE_KEY="your-supabase-service-role-or-secret-key"
SUPABASE_BUCKET="skincare-images"
SUPABASE_DB_URL="postgresql://postgres.your-project-ref:URL_ENCODED_PASSWORD@://supabase.com"

# CORS (comma-separated origins for frontend connection)
ALLOWED_ORIGINS="http://localhost:3000"
```

### 4. Initialize Supabase Database Schema
Run the following SQL in your **Supabase SQL Editor**:
```sql
CREATE TABLE IF NOT EXISTS chat_threads (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id TEXT NOT NULL,
    thread_id TEXT UNIQUE NOT NULL,
    title TEXT DEFAULT 'New Skincare Chat',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_threads_user_id ON chat_threads(user_id);
```
> *Note: LangGraph checkpoint tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) are provisioned automatically on server startup via `memory.setup()`.*

---

##  Running Locally

```bash
python app.py
```
- **API Base URL:** `http://localhost:8000/api`
- **Interactive Swagger UI:** `http://localhost:8000/docs`

---

##  API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Standard synchronous multimodal chat (text + optional image file) |
| `POST` | `/api/chat/stream` | Real-time SSE streaming chat with vision badge & token output |
| `GET` | `/api/threads/{user_id}` | List all active consultation threads for a specific patient |
| `GET` | `/api/history/{user_id}/{thread_id}` | Load saved LangGraph message history to resume a consultation |
| `DELETE` | `/api/threads/{user_id}/{thread_id}` | Permanently delete thread metadata, checkpoints, and storage images |

---

##  Docker Deployment

```bash
# Build production image
docker build -t derma-rag-backend:latest .

# Run container with environment variables
docker run -d -p 8000:8000 --env-file .env --name derma-backend derma-rag-backend:latest
```

---
