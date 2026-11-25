"""
MCP Tool Schemas for OpsMind Agentic Research System.
Provides strict Pydantic schemas for input validation and MCP tool registry discovery.
"""

from typing import Optional
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input parameters for external web research tool."""
    query: str = Field(
        ...,
        description="Search query string to search for recent articles, documentation, or external benchmarks."
    )
    num_results: Optional[int] = Field(
        None,
        description="Maximum number of search results to return (defaults to WEB_TOP_K, capped at MAX_TOOL_RESULTS)."
    )
    recent: Optional[bool] = Field(
        False,
        description="Restrict results to the last WEB_RECENT_TIME_RANGE (for time-sensitive questions)."
    )


class KnowledgeSearchInput(BaseModel):
    """Input parameters for internal ChromaDB vector knowledge search."""
    query: str = Field(
        ...,
        description="Semantic search query describing the technical concept, architecture, or algorithm."
    )
    category: Optional[str] = Field(
        None,
        description="Optional filter by category (e.g., 'evaluation', 'architecture', 'infrastructure', 'retrieval', 'security', 'prompting')."
    )
    limit: Optional[int] = Field(
        None,
        description="Number of relevant document chunks to retrieve (defaults to RAG_TOP_K, capped at MAX_TOOL_RESULTS)."
    )


class DocumentLookupInput(BaseModel):
    """Input parameters for direct document retrieval by document ID."""
    doc_id: str = Field(
        ...,
        description="Unique identifier of the knowledge document (e.g., 'doc-rag-eval-01', 'doc-agentic-rag-02')."
    )


# MCP Protocol Schema definitions for MCP Client Discovery
MCP_TOOL_DEFINITIONS = [
    {
        "name": "web_search",
        "description": "Searches the web for recent technical articles, documentation, framework updates, and external benchmarks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Max results to return (defaults to WEB_TOP_K)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "knowledge_search",
        "description": "Performs dense semantic vector similarity search in the local ChromaDB technical knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Semantic query describing the technical concept or question"
                },
                "category": {
                    "type": "string",
                    "description": "Category filter ('evaluation', 'architecture', 'infrastructure', 'retrieval', 'security', 'prompting')",
                    "nullable": True
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of chunks to return (defaults to RAG_TOP_K)"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "document_lookup",
        "description": "Looks up the full content and metadata of a specific internal knowledge document by its ID.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Unique document ID (e.g., 'doc-rag-eval-01')"
                }
            },
            "required": ["doc_id"]
        }
    }
]
