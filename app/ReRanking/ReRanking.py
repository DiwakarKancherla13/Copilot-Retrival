import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from app.Observability.LangfuseTracer import get_langfuse


load_dotenv()


class ReRanker:

    def __init__(self):
        self.model_name = os.getenv(
            "RERANKER_MODEL",
            "BAAI/bge-reranker-base"
        )

        self.device = os.getenv(
            "RERANKER_DEVICE",
            "cpu"
        )

        self.top_k = int(
            os.getenv(
                "RERANKER_TOP_K",
                "6"
            )
        )

        self.langfuse = get_langfuse()

        self.model = CrossEncoder(
            self.model_name,
            device=self.device
        )

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if not chunks:
            return []

        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="reranking"
        ) as span:

            span.update(
                input={
                    "query": query,
                    "input_chunk_count": len(chunks),
                    "chunks": chunks
                },
                metadata={
                    "model": self.model_name,
                    "device": self.device,
                    "top_k": self.top_k
                }
            )

            pairs = [
                (
                    query,
                    chunk.get("text", "")
                )
                for chunk in chunks
            ]

            scores = self.model.predict(
                pairs,
                show_progress_bar=False
            )

            reranked_chunks = []

            for chunk, score in zip(
                chunks,
                scores
            ):
                result = chunk.copy()

                result["rerank_score"] = float(
                    score
                )

                reranked_chunks.append(
                    result
                )

            reranked_chunks.sort(
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            top_chunks = reranked_chunks[
                :self.top_k
            ]

            span.update(
                output={
                    "output_chunk_count": len(top_chunks),
                    "reranked_chunks": top_chunks
                }
            )

            return top_chunks