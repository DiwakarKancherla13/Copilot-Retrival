from typing import List, Dict, Any

from app.ReRanking.ReRanking import ReRanker


reranker = ReRanker()


def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:

    if not query or not query.strip():
        raise ValueError(
            "Query cannot be empty"
        )

    if not chunks:
        return {
            "reranked_chunks": []
        }

    results = reranker.rerank(
        query=query,
        chunks=chunks
    )

    return {
        "reranked_chunks": results
    }