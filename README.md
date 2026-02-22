# RAG Agent API

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)


This is a production-ready backend built with **FastAPI** and **LangChain** that implements a multi-step GenAI workflow. It allows users to upload PDF documents, process them into a vector store, and perform intelligent queries using an **Agentic RAG** pattern.
---

**🌐 Deployed Backend Swagger Endpoint:** https://mohdaqib147-rag-agent-document-qna.hf.space/docs
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


## 🚀 GenAI Workflow Overview

The application goes beyond simple API calls by orchestrating a multi-step **Agentic Retrieval-Augmented Generation** workflow:

1. **Ingestion Pipeline**:
* PDFs are parsed using `PyPDFLoader`.
* Text is split into semantic chunks via `RecursiveCharacterTextSplitter` (1000 tokens/chunk, 200 overlap).
* Embeddings are generated using OpenAI's `text-embedding-3-small` and persisted in a **ChromaDB** vector store.


2. **Agentic Query Loop**:
* Instead of a basic search, the system uses a **LangGraph** state machine.
* The LLM (`gpt-4o`) acts as an agent that decides whether it needs to call a `retriever_tool` to answer the user's question.
* The agent can iterate: it can search, evaluate the retrieved context, and decide to search again if more information is needed before providing a final grounded response.



---

## 🛠️ Tech Stack

* **Framework**: FastAPI
* **Orchestration**: LangChain & LangGraph
* **LLM**: OpenAI GPT-4o
* **Vector Database**: ChromaDB
* **Environment**: Python 3.10+

---

## 💻 Local Setup Instructions

### 1. Prerequisites

Ensure you have Python installed and an OpenAI API Key ready.

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/pRiMeXoMeGa/rag_langgraph_doc_qna
cd rag_langgraph_doc_qna

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Environment Variables

Create a `.env` file in the root directory and add the following:

```env
OPENAI_API_KEY=your_openai_api_key_here
UPLOAD_DIR=uploads
VECTOR_STORE_DIR=vector_stores
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-small

```

### 4. Running the Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000

```

The API will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

---

## 📡 API Endpoints Summary

* `GET /health`: Check system status.
* `POST /upload`: Upload a PDF and trigger the embedding pipeline.
* `POST /query`: Send a question to the Agentic RAG regarding a specific `document_id`.
* `GET /documents`: List all processed documents and metadata.
* `DELETE /documents/{id}`: Remove a document and its associated vector embeddings.

---

## 🛡️ Quality & Safety Measures

* **Input Validation**: Strict file-type checking to ensure only PDFs are processed.
* **Grounded Responses**: The system prompt explicitly instructs the agent to answer based on the retrieved document context.
* **State Management**: Using LangGraph ensures the agent doesn't enter infinite loops and maintains a clear execution path.
* **Error Handling**: Comprehensive try-except blocks manage API failures and file system errors, providing clear HTTP exceptions to the frontend.

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

## 📖 API Documentation

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload a PDF document |
| `POST` | `/query` | Query a document |
| `GET` | `/documents` | List all documents |
| `DELETE` | `/documents/{id}` | Delete a document |

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

### 4. LLM: GPT-4o

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
You are an intelligent AI assistant who answers questions based on the document: "{doc_info['filename']}".
Use the retriever tool available to search for information in the document. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
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

### Platform Compatibility Summary

| Feature | Windows | macOS | Linux | Docker |
|---------|---------|-------|-------|--------|
| Upload | ✅ | ✅ | ✅ | ✅ |
| Query | ✅ | ✅ | ✅ | ✅ |
| List | ✅ | ✅ | ✅ | ✅ |
| Delete | ❌ | ✅ | ✅ | ✅ |

---

## 📝 License

MIT License - feel free to use this project for any purpose.

---



**Built with ❤️ using FastAPI, LangChain, LangGraph, React, and OpenAI**






