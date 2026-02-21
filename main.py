import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from schemas import (
    UploadResponse, 
    QueryRequest, 
    QueryResponse,
    DocumentListResponse,
    DeleteResponse,
    HealthResponse
)
from rag_agent import rag_manager

app = FastAPI(
    title="RAG Agent API",
    description="API for uploading PDFs and querying them using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the API is running."""
    return HealthResponse(status="healthy", message="RAG Agent API is running")


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and create embeddings.
    
    - **file**: PDF file to upload
    """
    # validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # save uploaded file
    file_path = os.path.join(settings.UPLOAD_DIR, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # process pdf and create embeddings
    try:
        result = rag_manager.process_pdf(file_path, file.filename)
        
        return UploadResponse(
            success=True,
            message="PDF uploaded and processed successfully",
            document_id=result["document_id"],
            filename=result["filename"],
            pages_count=result["pages_count"],
            chunks_count=result["chunks_count"]
        )
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Query a document using the RAG agent.
    
    - **query**: The question to ask
    - **document_id**: The ID of the document to query
    """
    try:
        result = rag_manager.query_document(request.document_id, request.query)
        
        return QueryResponse(
            success=True,
            answer=result["answer"],
            sources=result["sources"],
            tool_calls_made=result["tool_calls_made"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")


@app.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    documents = rag_manager.list_documents()
    return DocumentListResponse(documents=documents)


@app.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(document_id: str):
    """Delete a document and its vector store."""
    success = rag_manager.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DeleteResponse(success=True, message="Document deleted successfully")
