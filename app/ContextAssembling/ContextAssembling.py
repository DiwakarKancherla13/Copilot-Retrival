import os
from typing import List, Dict, Any

import tiktoken
from dotenv import load_dotenv

from app.Observability.LangfuseTracer import get_langfuse


load_dotenv()


class ContextAssembler:

    def __init__(self):

        self.max_tokens = int(
            os.getenv(
                "CONTEXT_MAX_TOKENS",
                "6000"
            )
        )

        self.citation_start = int(
            os.getenv(
                "CONTEXT_CITATION_START",
                "1"
            )
        )

        self.langfuse = get_langfuse()

        self.encoding = tiktoken.get_encoding(
            "o200k_base"
        )

    def assemble(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        with self.langfuse.start_as_current_observation(
            as_type="span",
            name="context-assembly"
        ) as span:

            span.update(
                input={
                    "input_chunk_count": len(chunks),
                    "chunks": chunks
                },
                metadata={
                    "max_tokens": self.max_tokens,
                    "tokenizer": "o200k_base"
                }
            )

            if not chunks:

                result = {
                    "context": "",
                    "chunks": [],
                    "citations": [],
                    "token_count": 0
                }

                span.update(
                    output=result
                )

                return result

            selected_chunks = []
            citations = []
            context_parts = []

            current_tokens = 0
            citation_number = self.citation_start

            for chunk in chunks:

                text = chunk.get(
                    "text",
                    ""
                ).strip()

                if not text:
                    continue

                chunk_tokens = self._count_tokens(
                    text
                )

                if (
                    current_tokens + chunk_tokens
                    > self.max_tokens
                ):
                    break

                citation = (
                    f"[{citation_number}]"
                )

                context_parts.append(
                    f"{citation} {text}"
                )

                citation_data = {
                    "citation": citation,
                    "chunk_id": chunk.get(
                        "chunk_id"
                    ),
                    "source": chunk.get(
                        "source"
                    ),
                    "page": chunk.get(
                        "page"
                    ),
                    "rerank_score": chunk.get(
                        "rerank_score"
                    )
                }

                citations.append(
                    citation_data
                )

                selected_chunk = chunk.copy()

                selected_chunk[
                    "citation"
                ] = citation

                selected_chunks.append(
                    selected_chunk
                )

                current_tokens += chunk_tokens

                citation_number += 1

            result = {
                "context": "\n\n".join(
                    context_parts
                ),
                "chunks": selected_chunks,
                "citations": citations,
                "token_count": current_tokens
            }

            span.update(
                output={
                    "selected_chunk_count": len(
                        selected_chunks
                    ),
                    "citation_count": len(
                        citations
                    ),
                    "token_count": current_tokens,
                    "context": result["context"],
                    "citations": citations
                }
            )

            return result

    def _count_tokens(
        self,
        text: str
    ) -> int:

        return len(
            self.encoding.encode(text)
        )