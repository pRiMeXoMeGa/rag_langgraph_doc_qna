import os
import re
import uuid
import json
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Annotated, Sequence

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from config import settings


SOURCE_PREVIEW_CHARS = 500


def safe_filename(filename: str) -> str:
    """Strip directories and unsafe characters from a client-supplied filename."""
    name = os.path.basename((filename or "").replace("\\", "/"))
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name or "document.pdf"


@tool
def retriever_tool(query: str) -> str:
    """
    Search and return information from the uploaded document.
    """
    # Schema only: the graph's tool node runs the search against the right document.
    raise NotImplementedError


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    document_id: str


class RAGAgentManager:
    """Manages per-document vector stores and one shared RAG agent graph."""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)
        self.llm_with_tools = self.llm.bind_tools([retriever_tool])
        self.document_stores: Dict[str, Dict[str, Any]] = {}
        self.metadata_file = os.path.join(settings.VECTOR_STORE_DIR, "metadata.json")
        self._load_metadata()
        # In-memory conversation history, keyed by thread (document + session).
        self.agent = self._build_graph().compile(checkpointer=MemorySaver())

    # ---------- metadata ----------

    def _load_metadata(self):
        """Load existing document metadata from disk."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                self.document_stores = json.load(f)

    def _save_metadata(self):
        """Save document metadata to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.document_stores, f, indent=2)

    # ---------- ingestion ----------

    @staticmethod
    def new_document_id() -> str:
        return uuid.uuid4().hex[:8]

    def process_pdf(self, file_path: str, filename: str, document_id: str) -> Dict[str, Any]:
        """Process a PDF file and create embeddings."""
        pages = PyPDFLoader(file_path).load()
        pages_count = len(pages)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(pages)
        chunks_count = len(chunks)

        persist_directory = os.path.join(settings.VECTOR_STORE_DIR, document_id)
        collection_name = f"doc_{document_id}"

        Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )

        self.document_stores[document_id] = {
            "filename": filename,
            "file_path": file_path,
            "persist_directory": persist_directory,
            "pages_count": pages_count,
            "chunks_count": chunks_count,
            "uploaded_at": datetime.now().isoformat(),
            "collection_name": collection_name
        }
        self._save_metadata()

        return {
            "document_id": document_id,
            "filename": filename,
            "pages_count": pages_count,
            "chunks_count": chunks_count
        }

    # ---------- retrieval ----------

    def get_retriever(self, document_id: str):
        """Get a retriever for a specific document."""
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

    def _search(self, document_id: str, query: str) -> str:
        docs = self.get_retriever(document_id).invoke(query)
        if not docs:
            return "No relevant information found in the document."
        return "\n\n".join(
            f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)
        )

    # ---------- agent graph (built once) ----------

    def _build_graph(self) -> StateGraph:
        def should_continue(state: AgentState):
            last = state["messages"][-1]
            return bool(getattr(last, "tool_calls", None))

        def call_llm(state: AgentState) -> Dict[str, Any]:
            filename = self.document_stores[state["document_id"]]["filename"]
            system_prompt = f"""
            You are an intelligent AI assistant who answers questions based on the document: "{filename}".
            Use the retriever tool available to search for information in the document. You can make multiple calls if needed.
            If you need to look up some information before asking a follow up question, you are allowed to do that!
            """
            messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
            return {"messages": [self.llm_with_tools.invoke(messages)]}

        def take_action(state: AgentState) -> Dict[str, Any]:
            results = []
            for t in state["messages"][-1].tool_calls:
                if t["name"] != retriever_tool.name:
                    result = "Incorrect Tool Name, Please Retry."
                else:
                    result = self._search(state["document_id"], t["args"].get("query", ""))
                results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
            return {"messages": results}

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
        return graph

    def query_document(self, document_id: str, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query a specific document using the RAG agent.

        Pass the same session_id on later calls to keep the conversation going;
        without one, every query starts a fresh conversation.
        """
        if document_id not in self.document_stores:
            raise ValueError(f"Document {document_id} not found")

        thread_id = f"{document_id}:{session_id or uuid.uuid4().hex}"
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)], "document_id": document_id},
            config={"configurable": {"thread_id": thread_id}}
        )

        # Only look at messages produced for this query, not earlier turns.
        messages = result["messages"]
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        tool_messages = [m for m in messages[start:] if isinstance(m, ToolMessage)]

        sources: List[str] = [
            m.content[:SOURCE_PREVIEW_CHARS] + "..." if len(m.content) > SOURCE_PREVIEW_CHARS else m.content
            for m in tool_messages
        ]

        return {
            "answer": messages[-1].content,
            "tool_calls_made": len(tool_messages),
            "sources": sources,
            "session_id": session_id
        }

    # ---------- management ----------

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

        if os.path.exists(doc_info["persist_directory"]):
            shutil.rmtree(doc_info["persist_directory"])

        if os.path.exists(doc_info["file_path"]):
            os.remove(doc_info["file_path"])

        del self.document_stores[document_id]
        self._save_metadata()

        return True


rag_manager = RAGAgentManager()
