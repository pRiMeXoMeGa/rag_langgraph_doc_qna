from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class UploadResponse(BaseModel):
    success: bool
    message: str
    document_id: str
    filename: str
    pages_count: int
    chunks_count: int

class QueryRequest(BaseModel):
    query: str
    document_id: str

class QueryResponse(BaseModel):
    success: bool
    answer: str
    tool_calls_made: int = 0

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    uploaded_at: str
    pages_count: int
    chunks_count: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]

class DeleteResponse(BaseModel):
    success: bool
    message: str

class HealthResponse(BaseModel):
    status: str
    message: str