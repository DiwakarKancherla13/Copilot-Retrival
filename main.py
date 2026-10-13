from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.MCP.ReRankingTool import rerank_chunks
from app.MCP.ContextAssemblingTool import assemble_context
from app.Observability.LangfuseTracer import get_langfuse


app = FastAPI(
    title="RAG Retrieval API",
    version="1.0.0"
)


langfuse = get_langfuse()


class RetrievalRequest(BaseModel):
    query: str
    chunks: List[Dict[str, Any]]


@app.get("/")
def root():
    return {
        "message": "RAG Retrieval API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/retrieval")
def retrieval(
    request: RetrievalRequest
):

    if not request.query.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    if not request.chunks:
        raise HTTPException(
            status_code=400,
            detail="Chunks cannot be empty"
        )

    try:

        with langfuse.start_as_current_observation(
            as_type="span",
            name="retrieval-request"
        ) as trace:

            trace.update(
                input={
                    "query": request.query,
                    "input_chunk_count": len(
                        request.chunks
                    ),
                    "chunks": request.chunks
                }
            )

            rerank_response = rerank_chunks(
                query=request.query,
                chunks=request.chunks
            )

            reranked_chunks = (
                rerank_response.get(
                    "reranked_chunks",
                    []
                )
            )

            context_response = (
                assemble_context(
                    chunks=reranked_chunks
                )
            )

            response = {
                "query": request.query,

                "reranked_chunks":
                    reranked_chunks,

                "context":
                    context_response.get(
                        "context",
                        ""
                    ),

                "citations":
                    context_response.get(
                        "citations",
                        []
                    ),

                "token_count":
                    context_response.get(
                        "token_count",
                        0
                    )
            }

            trace.update(
                output={
                    "reranked_chunk_count":
                        len(reranked_chunks),

                    "context_token_count":
                        response["token_count"],

                    "citation_count":
                        len(
                            response[
                                "citations"
                            ]
                        ),

                    "response":
                        response
                }
            )

            return response

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )