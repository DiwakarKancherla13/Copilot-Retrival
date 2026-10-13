from typing import List, Dict, Any

from app.ContextAssembling.ContextAssembling import ContextAssembler


context_assembler = ContextAssembler()


def assemble_context(
    chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:

    return context_assembler.assemble(
        chunks=chunks
    )