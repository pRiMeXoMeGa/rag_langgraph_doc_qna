import os
import uuid
import json
from datetime import datetime
from typing import Dict, Any, TypedDict, Annotated, Sequence
from operator import add as add_messages

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END

from config import settings


class RAGAgentManager:
    """Manages multiple RAG agents for different documents."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        self.document_stores: Dict[str, Dict[str, Any]] = {}
        self.metadata_file = os.path.join(settings.VECTOR_STORE_DIR, "metadata.json")
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing document metadata from disk."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.document_stores = json.load(f)
    
    def _save_metadata(self):
        """Save document metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.document_stores, f, indent=2)
    
    def process_pdf(self, file_path: str, filename: str) -> Dict[str, Any]:
        """Process a PDF file and create embeddings."""
        
        document_id = str(uuid.uuid4())[:8]
        
        pdf_loader = PyPDFLoader(file_path)
        pages = pdf_loader.load()
        pages_count = len(pages)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(pages)
        chunks_count = len(chunks)
        
        persist_directory = os.path.join(settings.VECTOR_STORE_DIR, document_id)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=f"doc_{document_id}"
        )
        
        self.document_stores[document_id] = {
            "filename": filename,
            "file_path": file_path,
            "persist_directory": persist_directory,
            "pages_count": pages_count,
            "chunks_count": chunks_count,
            "uploaded_at": datetime.now().isoformat(),
            "collection_name": f"doc_{document_id}"
        }
        self._save_metadata()
        
        return {
            "document_id": document_id,
            "filename": filename,
            "pages_count": pages_count,
            "chunks_count": chunks_count
        }
    
    def get_retriever(self, document_id: str):
        """Get or create a retriever for a specific document."""
        if document_id not in self.document_stores:
            raise ValueError(f"Document {document_id} not found")
        
        doc_info = self.document_stores[document_id]
        
        vectorstore = Chroma(
            persist_directory=doc_info["persist_directory"],
            embedding_function=self.embeddings,
            collection_name=doc_info["collection_name"]
        )
        
        return vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.RETRIEVER_K}
        )
    
    def query_document(self, document_id: str, query: str) -> Dict[str, Any]:
        """Query a specific document using the RAG agent."""
        
        retriever = self.get_retriever(document_id)
        doc_info = self.document_stores[document_id]
        
        @tool
        def retriever_tool(query: str) -> str:
            """
            Search and return information from the uploaded document.
            """
            docs = retriever.invoke(query)
            
            if not docs:
                return "No relevant information found in the document."
            
            results = []
            for i, doc in enumerate(docs):
                results.append(f"Document {i+1}:\n{doc.page_content}")
            
            return "\n\n".join(results)
        
        tools = [retriever_tool]
        tools_dict = {t.name: t for t in tools}
        
        llm_with_tools = self.llm.bind_tools(tools)
        
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], add_messages]
        
        system_prompt = f"""
        You are an intelligent AI assistant who answers questions based on the document: "{doc_info['filename']}".
        Use the retriever tool available to search for information in the document. You can make multiple calls if needed.
        If you need to look up some information before asking a follow up question, you are allowed to do that!
        """
        
        tool_calls_count = [0]
        
        def should_continue(state: AgentState):
            result = state['messages'][-1]
            return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
        
        def call_llm(state: AgentState) -> AgentState:
            messages = list(state['messages'])
            messages = [SystemMessage(content=system_prompt)] + messages
            message = llm_with_tools.invoke(messages)
            return {'messages': [message]}
        
        def take_action(state: AgentState) -> AgentState:
            tool_calls = state['messages'][-1].tool_calls
            results = []
            for t in tool_calls:
                tool_calls_count[0] += 1
                if t['name'] not in tools_dict:
                    result = "Incorrect Tool Name, Please Retry."
                else:
                    result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
                
                results.append(ToolMessage(
                    tool_call_id=t['id'],
                    name=t['name'],
                    content=str(result)
                ))
            
            return {'messages': results}
        
        graph = StateGraph(AgentState)
        graph.add_node("llm", call_llm)
        graph.add_node("retriever_agent", take_action)
        graph.add_conditional_edges(
            "llm",
            should_continue,
            {True: "retriever_agent", False: END}
        )
        graph.add_edge("retriever_agent", "llm")
        graph.set_entry_point("llm")
        
        rag_agent = graph.compile()
        
        messages = [HumanMessage(content=query)]
        result = rag_agent.invoke({"messages": messages})
        
        sources = []
        for msg in result['messages']:
            if isinstance(msg, ToolMessage):
                sources.append(msg.content[:500] + "..." if len(msg.content) > 500 else msg.content)
        
        return {
            "answer": result['messages'][-1].content,
            "tool_calls_made": tool_calls_count[0]
        }
    
    def list_documents(self) -> list:
        """List all uploaded documents."""
        return [
            {
                "document_id": doc_id,
                "filename": info["filename"],
                "uploaded_at": info["uploaded_at"],
                "pages_count": info["pages_count"],
                "chunks_count": info["chunks_count"]
            }
            for doc_id, info in self.document_stores.items()
        ]
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its vector store."""
        if document_id not in self.document_stores:
            return False
        
        doc_info = self.document_stores[document_id]
        
        import shutil
        if os.path.exists(doc_info["persist_directory"]):
            shutil.rmtree(doc_info["persist_directory"])
        
        if os.path.exists(doc_info["file_path"]):
            os.remove(doc_info["file_path"])
        
        del self.document_stores[document_id]
        self._save_metadata()
        
        return True

rag_manager = RAGAgentManager()