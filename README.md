# 📚 RAG Agent API

A production-ready Retrieval-Augmented Generation (RAG) API built with FastAPI and LangGraph that enables intelligent document querying using OpenAI's language models.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [Architecture](#-architecture)
- [Setup Instructions](#-setup-instructions)
- [API Documentation](#-api-documentation)
- [Frontend Integration](#-frontend-integration)
- [GenAI Workflow Design](#-genai-workflow-design)
- [Technical Decisions & Trade-offs](#-technical-decisions--trade-offs)
- [AI Limitations & Safety](#-ai-limitations-errors--safety-considerations)
- [Known Issues & Troubleshooting](#-known-issues--troubleshooting)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🎯 Problem Statement

### The Problem

Organizations and individuals deal with an ever-growing volume of PDF documents—research papers, legal contracts, technical manuals, financial reports, and more. Extracting specific information from these documents is:

| Challenge | Description |
|-----------|-------------|
| **Time-consuming** | Manually searching through hundreds of pages |
| **Error-prone** | Human fatigue leads to missed information |
| **Inefficient** | Same questions asked repeatedly require re-reading |
| **Inaccessible** | Non-technical users struggle with document analysis |

### Who It's For

| User Type | Use Case |
|-----------|----------|
| **Researchers** | Query academic papers, extract citations, compare findings |
| **Legal Professionals** | Search contracts, find specific clauses, review agreements |
| **Students** | Study from textbooks, prepare for exams, understand concepts |
| **Business Analysts** | Extract insights from reports, analyze financial documents |
| **Support Teams** | Query product manuals, find troubleshooting steps |
| **Developers** | Build document-powered chatbots and applications |

### The Solution

This RAG Agent API provides a simple REST interface to:
1. **Upload** PDF documents
2. **Process** them into searchable vector embeddings
3. **Query** them using natural language questions
4. **Receive** accurate, contextual answers

---

## ✨ Features

- 📄 **PDF Upload & Processing** - Automatic text extraction and chunking
- 🔍 **Semantic Search** - Find relevant content using meaning, not just keywords
- 🤖 **Agentic RAG** - LLM decides when and how to search documents
- 💾 **Persistent Storage** - Vector stores survive server restarts
- 📚 **Multi-Document Management** - Upload and manage multiple documents
- 🔌 **RESTful API** - Easy integration with any application
- 📊 **Health Monitoring** - Built-in health check endpoint

---

## 🏗 Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT                                   │
│                  (Web App / Mobile / CLI)                        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP/REST
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FASTAPI SERVER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ /upload  │  │ /query   │  │/documents│  │ /health      │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘    │
└───────┼─────────────┼─────────────┼─────────────────────────────┘
        │             │             │
        ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RAG AGENT MANAGER                             │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                    LangGraph Agent                       │    │
│  │  ┌─────────┐    ┌──────────────┐    ┌─────────────┐    │    │
│  │  │   LLM   │◄──►│   Retriever  │◄──►│   Tools     │    │    │
│  │  │ (GPT-4) │    │     Tool     │    │  Executor   │    │    │
│  │  └─────────┘    └──────────────┘    └─────────────┘    │    │
│  └─────────────────────────────────────────────────────────┘    │
└───────┼─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐      │
│  │   ChromaDB   │  │  PDF Files   │  │   Metadata JSON  │      │
│  │ Vector Store │  │   Storage    │  │     Storage      │      │
│  └──────────────┘  └──────────────┘  └──────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- OpenAI API key
- pip (Python package manager)

### Step 1: Clone the Repository

```bash
git clone [https://github.com/yourusername/rag-agent-api.git](https://github.com/yourusername/rag-agent-api.git)
cd rag-agent-api
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```txt
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
pydantic>=2.0.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
langchain-chroma>=0.1.0
langgraph>=0.0.20
chromadb>=0.4.0
pypdf>=3.15.0
python-dotenv>=1.0.0
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here

# Model Configuration
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

# RAG Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVER_K=4

# Storage Paths
UPLOAD_DIR=./uploads
VECTOR_STORE_DIR=./vector_stores
```

### Step 5: Create config.py

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", 4))
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    VECTOR_STORE_DIR: str = os.getenv("VECTOR_STORE_DIR", "./vector_stores")
    
    def __init__(self):
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_DIR, exist_ok=True)

settings = Settings()
```

### Step 6: Run the Server

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Step 7: Verify Installation

```bash
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","message":"RAG Agent API is running"}
```

### Step 8: Access API Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 📖 API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PDF document |
| `POST` | `/query` | Query a document |
| `GET` | `/documents` | List all documents |
| `DELETE` | `/documents/{id}` | Delete a document |

### Upload a PDF

```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/document.pdf"
```

**Response:**
```json
{
  "success": true,
  "message": "PDF uploaded and processed successfully",
  "document_id": "a1b2c3d4",
  "filename": "document.pdf",
  "pages_count": 25,
  "chunks_count": 48
}
```

### Query a Document

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main conclusion of this document?",
    "document_id": "a1b2c3d4"
  }'
```

**Response:**
```json
{
  "success": true,
  "answer": "The main conclusion of the document is...",
  "tool_calls_made": 2
}
```

### List All Documents

```bash
curl -X GET "http://localhost:8000/documents"
```

### Delete a Document

```bash
curl -X DELETE "http://localhost:8000/documents/a1b2c3d4"
```

> ⚠️ **Note:** Delete functionality has limitations on Windows. See [Known Issues](#-known-issues--troubleshooting).

---

## 💻 Frontend Integration

When connecting a frontend client, this API maps cleanly to standard REST patterns. Here is an implementation example using React and TypeScript to interact with the `/query` endpoint:

```typescript
// types.ts
export interface QueryRequest {
  query: string;
  document_id: string;
}

export interface QueryResponse {
  success: boolean;
  answer: string;
  tool_calls_made: number;
}

// api.ts
export const queryDocument = async (data: QueryRequest): Promise<QueryResponse> => {
  const response = await fetch('http://localhost:8000/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    throw new Error('Failed to query document');
  }

  return response.json();
};
```

---

## 🧠 GenAI Workflow Design

### Why Agentic RAG over Simple RAG?

I chose an **Agentic RAG** architecture using **LangGraph** over a simple retrieve-and-generate pipeline:

```text
SIMPLE RAG                          AGENTIC RAG (Our Approach)
==========                          ==========================

User Query                          User Query
    │                                   │
    ▼                                   ▼
┌─────────┐                        ┌─────────┐
│Retrieve │ (always)               │   LLM   │ (decides)
└────┬────┘                        └────┬────┘
     │                                  │
     ▼                                  ▼
┌─────────┐                        ┌─────────────┐
│   LLM   │                        │ Need info?  │
└────┬────┘                        └──────┬──────┘
     │                                Yes │ No
     ▼                                 │  └──► Direct Answer
  Answer                               ▼
                                  ┌─────────┐
                                  │Retrieve │
                                  └────┬────┘
                                       │
                                       ▼
                                  ┌─────────┐
                                  │   LLM   │◄──┐
                                  └────┬────┘   │
                                       │        │
                                  ┌────▼────┐   │
                                  │ Enough? │───┘
                                  └────┬────┘ No
                                   Yes │
                                       ▼
                                    Answer
```

### Design Rationale

| Aspect | Simple RAG | Agentic RAG (Chosen) |
|--------|-----------|---------------------|
| **Retrieval Control** | Always retrieves | LLM decides when to retrieve |
| **Multi-hop Questions** | Single retrieval | Multiple retrievals possible |
| **Query Refinement** | No | LLM can reformulate queries |
| **Efficiency** | May retrieve unnecessarily | Only retrieves when needed |
| **Complex Queries** | Limited | Handles follow-up reasoning |

### Why LangGraph?

| Benefit | Description |
|---------|-------------|
| **Explicit Control Flow** | Visual, debuggable state machine |
| **Flexibility** | Easy to add new tools and nodes |
| **State Management** | Built-in message history handling |
| **Production Ready** | Better error handling than chains |
| **Extensibility** | Can add human-in-the-loop, branching logic |

---

## ⚖️ Technical Decisions & Trade-offs

### 1. Vector Database: ChromaDB

| Aspect | Details |
|--------|---------|
| **Chose Over** | Pinecone, Weaviate, Milvus, FAISS |
| **Why** | Zero configuration, embedded, persistent, free |
| **Trade-off** | Less scalable than cloud solutions |
| **Good For** | Prototypes, small-medium deployments |
| **Limitation** | Single-node only, no distributed support |

### 2. Embedding Model: text-embedding-3-small

| Aspect | Details |
|--------|---------|
| **Chose Over** | text-embedding-ada-002, text-embedding-3-large |
| **Why** | Best cost-performance ratio, 1536 dimensions |
| **Trade-off** | Slightly less accurate than large model |
| **Cost** | $0.02 per 1M tokens vs $0.13 for large |

### 3. Chunking Strategy

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **chunk_size** | 1000 | Balances context vs specificity |
| **chunk_overlap** | 200 | Prevents information loss at boundaries |

### 4. LLM: GPT-4o-mini

| Aspect | Details |
|--------|---------|
| **Chose Over** | GPT-4, GPT-3.5-turbo, Claude, Llama |
| **Why** | Best balance of capability, speed, and cost |
| **Trade-off** | Less capable than full GPT-4 |
| **Cost** | ~20x cheaper than GPT-4 |

### 5. Document Storage Strategy

```text
vector_stores/
├── abc123/          # Document 1
│   └── chroma.sqlite3
├── def456/          # Document 2
│   └── chroma.sqlite3
└── metadata.json    # Central registry
```

| Approach | Pros | Cons |
|----------|------|------|
| **Per-Document (Chosen)** | Isolation, easy deletion | No cross-doc search |
| **Single Store** | Cross-document search | Harder deletion |

---

## 🛡 AI Limitations, Errors & Safety Considerations

### 1. Hallucination Mitigation

**Strategies Implemented:**
- ✅ Explicit grounding instructions in system prompt
- ✅ Tool-based retrieval (LLM must use retriever)
- ✅ Temperature set to 0 (deterministic outputs)
- ✅ System prompt instructs to say "not found" when information unavailable

```python
system_prompt = """
You are an AI assistant who answers questions based ONLY on 
the document. Use the retriever tool to search for information.
If information is not found, say so clearly.
DO NOT make up information.
"""
```

### 2. Error Handling

| Error Type | Handling |
|------------|----------|
| Invalid file type | 400 Bad Request |
| Document not found | 404 Not Found |
| Processing failure | 500 with cleanup |
| OpenAI API errors | Graceful error messages |

### 3. Safety Considerations

| Risk | Mitigation |
|------|------------|
| **Prompt Injection** | System prompts are server-side only |
| **Data Leakage** | Documents isolated per document_id |
| **Malicious PDFs** | PyPDF handles parsing safely |
| **API Abuse** | CORS configured (add rate limiting for production) |

### 4. Current Limitations

| Limitation | Impact | Workaround |
|------------|--------|------------|
| Single document query only | Can't compare across docs | Query each separately |
| No OCR support | Scanned PDFs won't work | Use OCR-processed PDFs |
| No authentication | Anyone can access | Add JWT/API keys |
| Windows delete issues | Can't delete on Windows | Use Docker/Linux |

---

## 🐛 Known Issues & Troubleshooting

### Delete Functionality - Windows Issue

| Platform | Delete Works? |
|----------|---------------|
| **Windows (Local)** | ❌ No |
| **Linux/macOS** | ✅ Yes |
| **Deployed (Linux servers)** | ✅ Yes |
| **Docker** | ✅ Yes |

#### The Problem

On Windows, deleting documents fails with:

```text
PermissionError: [WinError 32] The process cannot access the file 
because it is being used by another process
```

#### Root Cause

**Windows file locking behavior:**

```text
WINDOWS                              LINUX/UNIX
=======                              ==========

ChromaDB opens file                  ChromaDB opens file
       │                                    │
       ▼                                    ▼
┌─────────────┐                     ┌─────────────┐
│   File      │                     │   File      │
│  🔒 LOCKED  │                     │ 🔓 UNLOCKED │
│Cannot delete│                     │ Can delete  │
└─────────────┘                     └─────────────┘
```

- **Windows**: Uses mandatory file locking. Files cannot be deleted while open.
- **Linux/macOS**: Uses reference counting. Files can be unlinked while open.

#### Solutions

**Option 1: Use Docker (Recommended for Windows)**

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p uploads vector_stores
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t rag-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key rag-api
```

**Option 2: Use WSL2 on Windows**

```bash
# Run in WSL2 terminal
cd /mnt/c/path/to/project
source venv/bin/activate
uvicorn main:app --reload
```

**Option 3: Manual Cleanup**

```bash
# Stop the server (Ctrl+C)
# Manually delete: vector_stores/<document_id>/
# Restart the server
```

### Platform Compatibility Summary

| Feature | Windows | macOS | Linux | Docker |
|---------|---------|-------|-------|--------|
| Upload | ✅ | ✅ | ✅ | ✅ |
| Query | ✅ | ✅ | ✅ | ✅ |
| List | ✅ | ✅ | ✅ | ✅ |
| Delete | ❌ | ✅ | ✅ | ✅ |

---

## 🚧 Future Improvements

### High Priority

| Improvement | Description | Effort |
|-------------|-------------|--------|
| **Multi-document queries** | Query across multiple PDFs simultaneously | Medium |
| **Authentication** | JWT tokens, API keys, user management | Medium |
| **Rate limiting** | Prevent API abuse | Low |
| **Streaming responses** | Real-time token streaming | Medium |
| **Fix Windows delete** | Proper connection cleanup | Medium |

### Medium Priority

| Improvement | Description | Effort |
|-------------|-------------|--------|
| **OCR support** | Handle scanned PDFs | Medium |
| **Multiple file formats** | DOCX, TXT, HTML, Markdown | Medium |
| **Conversation memory** | Multi-turn conversations | Medium |
| **Hybrid search** | Semantic + keyword (BM25) | Medium |
| **Caching** | Cache frequent queries | Low |

### Nice to Have

| Improvement | Description | Effort |
|-------------|-------------|--------|
| **Admin dashboard** | Web UI for management | High |
| **Analytics** | Query tracking, performance metrics | Medium |
| **Webhooks** | Notify on processing complete | Low |
| **Batch upload** | Multiple files at once | Low |
| **Export conversations** | Download Q&A history | Low |

### Scalability Roadmap

```text
Current                             Future
=======                             ======

┌─────────────┐                    ┌──────────────┐
│  FastAPI    │                    │Load Balancer │
│  (Single)   │                    └──────┬───────┘
└──────┬──────┘                           │
       │                         ┌────────┼────────┐
       ▼                         ▼        ▼        ▼
┌─────────────┐              ┌──────┐ ┌──────┐ ┌──────┐
│  ChromaDB   │              │ API  │ │ API  │ │ API  │
│  (Local)    │              │  1   │ │  2   │ │  3   │
└─────────────┘              └──────┘ └──────┘ └──────┘
                                       │
                                       ▼
                              ┌───────────────┐
                              │   Pinecone/   │
                              │   Weaviate    │
                              └───────────────┘
```

---

## 📁 Project Structure

```text
rag-agent-api/
├── main.py              # FastAPI application & endpoints
├── rag_agent.py         # RAG Agent Manager & LangGraph logic
├── schemas.py           # Pydantic models
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── .env                 # Environment variables (create this)
├── README.md            # This file
├── uploads/             # Uploaded PDF storage
└── vector_stores/       # ChromaDB vector stores
    ├── <doc_id>/        # Per-document vector store
    └── metadata.json    # Document registry
```

---

## 📝 License

MIT License - feel free to use this project for any purpose.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Support

For questions or issues:
- Open a GitHub issue
- Check existing documentation
- Review the troubleshooting section

---

**Built with ❤️ using FastAPI, LangChain, LangGraph, React, and OpenAI**
