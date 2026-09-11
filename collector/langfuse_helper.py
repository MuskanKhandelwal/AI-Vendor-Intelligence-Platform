"""Thin wrapper around the Langfuse SDK for tracing LLM calls in the collector layer."""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def get_client():
    """Return the global Langfuse client, or None if credentials are not configured.

    Uses Langfuse v3+ idiomatic pattern: reads credentials from environment variables.
    Returns None if LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set.
    """
    try:
        from langfuse import get_client as langfuse_get_client

        # Check credentials are configured
        if not os.getenv("LANGFUSE_PUBLIC_KEY") or not os.getenv("LANGFUSE_SECRET_KEY"):
            return None

        # Use Langfuse's built-in get_client() which manages the global singleton
        return langfuse_get_client()
    except ImportError:
        # Fallback for older versions that don't have get_client
        print("WARNING: Langfuse get_client() not available, using fallback initialization")
        return _fallback_client()
    except Exception as exc:
        print(f"WARNING: Could not initialise Langfuse client: {exc}")
        return None


def _fallback_client():
    """Fallback initialization for older Langfuse versions."""
    try:
        from langfuse import Langfuse

        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if not public_key or not secret_key:
            return None

        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
    except Exception as exc:
        print(f"WARNING: Fallback Langfuse initialization failed: {exc}")
        return None


def trace_llm_call(
    trace_name: str,
    input_text: str,
    output_text: str,
    model_name: str,
    metadata: dict[str, Any] | None = None,
    usage_details: dict[str, int] | None = None,
    model_parameters: dict[str, Any] | None = None,
) -> str | None:
    """Create a Langfuse trace with a single generation span.

    `input_text` and `output_text` must be the actual prompt sent and the
    actual text returned. Passing a paraphrase makes the trace useless for the
    one job it exists to do: explaining why the model answered as it did.

    Returns the trace_id string, or None if Langfuse is not configured.
    """
    client = get_client()
    if client is None:
        return None

    metadata = metadata or {}

    try:
        observation = client.start_observation(
            name=trace_name,
            as_type="generation",
            model=model_name,
            input=input_text,
            output=output_text,
            metadata=metadata,
            usage_details=usage_details,
            model_parameters=model_parameters,
        )

        observation.end()
        client.flush()
        return observation.trace_id

    except Exception as exc:
        print(f"WARNING: Langfuse trace failed ({trace_name}): {exc}")
        return None


def trace_collector_call(
    trace_name: str,
    company_name: str,
    prompt: str,
    response_text: str,
    model_name: str,
    response=None,
    **extra_metadata: Any,
) -> str | None:
    """Trace a collector LLM call using the real prompt and real response.

    Replaces an earlier `trace_signal_classification` that reconstructed both
    sides from a few scalars, e.g. input "Classify the following headline for
    X" and output "signal_type=funding, importance_score=85". Neither string
    was ever sent to or returned by the model, so a trace could not be used to
    debug a misclassification -- it did not contain the prompt or the answer.

    Pass the Groq `response` object to record real token usage and sampling
    parameters alongside it.

    Returns the trace_id string, or None if Langfuse is not configured.
    """
    usage_details = None
    model_parameters = None
    if response is not None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            usage_details = {
                "input": getattr(usage, "prompt_tokens", 0),
                "output": getattr(usage, "completion_tokens", 0),
                "total": getattr(usage, "total_tokens", 0),
            }

    return trace_llm_call(
        trace_name=trace_name,
        input_text=prompt,
        output_text=response_text,
        model_name=model_name,
        metadata={"company_name": company_name, **extra_metadata},
        usage_details=usage_details,
        model_parameters=model_parameters,
    )


if __name__ == "__main__":
    client = get_client()
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if client is None:
        print(
            "Langfuse client is not configured.\n"
            "Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in your .env file.\n"
            f"  LANGFUSE_PUBLIC_KEY : {'set' if public_key else 'MISSING'}\n"
            f"  LANGFUSE_SECRET_KEY : {'set' if secret_key else 'MISSING'}\n"
            f"  LANGFUSE_HOST       : {host}"
        )
    else:
        trace_id = trace_llm_call(
            trace_name="langfuse-connection-test",
            input_text="Is the Langfuse integration working?",
            output_text="Yes — test trace received successfully.",
            model_name="test",
            metadata={"source": "langfuse_helper.__main__"},
        )
        if trace_id:
            print(f"Langfuse connection OK. Test trace ID: {trace_id}")
            print(f"View it at: {host}/traces/{trace_id}")
        else:
            print("Langfuse client initialised but trace submission failed.")
            print("Check the WARNING messages above for details.")
