from app.Observability.LangfuseTracer import get_langfuse


langfuse = get_langfuse()


if langfuse.auth_check():
    print("Langfuse authentication successful")
else:
    print("Langfuse authentication failed")


with langfuse.start_as_current_observation(
    as_type="span",
    name="test-retrieval"
) as span:

    span.update(
        input={
            "query": "What is the loan approval process?"
        }
    )

    result = {
        "status": "success",
        "message": "Langfuse test trace"
    }

    span.update(
        output=result
    )


langfuse.flush()

print("Trace sent to Langfuse")