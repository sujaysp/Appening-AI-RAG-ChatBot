from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import build_rag_graph


app = FastAPI(
    title="Agentic AI RAG Chatbot",
    description=(
        "A strict document-grounded RAG chatbot built with "
        "LangGraph, Pinecone, local embeddings, and Gemini."
    ),
    version="1.0.0",
)

graph = build_rag_graph()


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the Agentic AI eBook",
    )


class ChatResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


@app.get("/")
def root():
    return {
        "name": "Agentic AI RAG Chatbot",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = graph.invoke(
            {
                "question": request.query,
            }
        )

        return ChatResponse(
            query=request.query,
            final_answer=result.get(
                "answer",
                "I cannot answer this based on the provided eBook.",
            ),
            retrieved_context_chunks=result.get(
                "context",
                [],
            ),
            confidence_score=float(
                result.get(
                    "confidence_score",
                    0.0,
                )
            ),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG pipeline error: {str(exc)}",
        ) from exc