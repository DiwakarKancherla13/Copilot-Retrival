import asyncio
import json
import traceback

import app.MCP.server as mcp_server


SAMPLE_CHUNKS = [
    {
        "chunk_id": "1",
        "text": "Loan approval requires KYC, income checks, and credit score verification.",
        "source": "policy.pdf",
        "page": 2,
    },
    {
        "chunk_id": "2",
        "text": "Disbursement happens after sanction and agreement signing.",
        "source": "policy.pdf",
        "page": 5,
    },
]


def _log(title: str, payload=None) -> None:
    print("\n" + "=" * 80)
    print(title)
    if payload is not None:
        if isinstance(payload, (dict, list)):
            print(json.dumps(payload, indent=2))
        else:
            print(payload)
    print("=" * 80)


async def run_test() -> int:
    try:
        _log("STEP 1: Listing MCP tools")
        tools = await mcp_server.mcp.list_tools()
        tool_names = [tool.name for tool in tools]
        _log("Registered MCP tools", tool_names)

        required_tools = {"rerank_chunks", "assemble_context"}
        missing = required_tools - set(tool_names)
        if missing:
            _log("FAIL: Missing required tools", sorted(missing))
            return 1

        _log("PASS: Required tools are registered")

        _log("STEP 2: Calling rerank_chunks tool")
        rerank_result = await mcp_server.mcp.call_tool(
            "rerank_chunks",
            {
                "query": "loan approval process",
                "chunks": SAMPLE_CHUNKS,
            },
        )

        if rerank_result.is_error:
            _log("FAIL: rerank_chunks returned error", rerank_result)
            return 1

        reranked_chunks = (
            rerank_result.structured_content.get("result", {}).get("reranked_chunks", [])
        )
        _log("PASS: rerank_chunks returned response", rerank_result.structured_content)

        if not reranked_chunks:
            _log("FAIL: rerank_chunks returned empty reranked_chunks")
            return 1

        _log("STEP 3: Calling assemble_context tool")
        assemble_result = await mcp_server.mcp.call_tool(
            "assemble_context",
            {"chunks": reranked_chunks},
        )

        if assemble_result.is_error:
            _log("FAIL: assemble_context returned error", assemble_result)
            return 1

        assembled = assemble_result.structured_content.get("result", {})
        _log("PASS: assemble_context returned response", assembled)

        context = assembled.get("context", "")
        token_count = assembled.get("token_count", 0)
        citations = assembled.get("citations", [])

        if not context.strip():
            _log("FAIL: assemble_context returned empty context")
            return 1

        _log(
            "PASS: MCP tool invocation summary",
            {
                "reranked_chunk_count": len(reranked_chunks),
                "context_length": len(context),
                "token_count": token_count,
                "citation_count": len(citations),
            },
        )

        _log("FINAL RESULT: SUCCESS - ALL MCP TOOLS WORKING")
        return 0
    except Exception as exc:
        _log("FINAL RESULT: FAILURE - EXCEPTION RAISED", str(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run_test()))
