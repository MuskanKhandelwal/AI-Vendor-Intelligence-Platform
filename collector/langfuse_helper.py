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
) -> str | None:
    """Create a Langfuse trace with a single generation span.

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
        )

        observation.end()
        client.flush()
        return observation.trace_id

    except Exception as exc:
        print(f"WARNING: Langfuse trace failed ({trace_name}): {exc}")
        return None


def trace_signal_classification(
    company_name: str,
    headline: str,
    signal_type: str,
    importance_score: int,
    model_name: str,
) -> str | None:
    """Convenience wrapper for tracing a signal classification LLM call.

    Returns the trace_id string, or None if Langfuse is not configured.
    """
    input_text = f"Classify the following headline for {company_name}:\n\n{headline}"
    output_text = (
        f"signal_type={signal_type}, importance_score={importance_score}"
    )
    metadata = {
        "company_name": company_name,
        "signal_type": signal_type,
        "importance_score": importance_score,
    }

    return trace_llm_call(
        trace_name="signal-classification",
        input_text=input_text,
        output_text=output_text,
        model_name=model_name,
        metadata=metadata,
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
