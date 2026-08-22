"""Shared LLM initialization for agents."""

import os
from langchain_aws import ChatBedrockConverse
from langfuse.langchain import CallbackHandler
from dotenv import load_dotenv

load_dotenv()

# Langfuse callback (singleton, initialized once)
_langfuse_callback = None


def get_langfuse_callback():
    """Get or create Langfuse callback handler (v3+ idiomatic pattern).

    Uses environment variables: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST.
    Returns None if credentials not configured.
    """
    global _langfuse_callback

    if _langfuse_callback is not None:
        return _langfuse_callback

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not public_key or not secret_key:
        return None

    try:
        # Langfuse v3+ reads credentials (public/secret key, host) from the
        # environment via its global client — the handler takes no kwargs.
        _langfuse_callback = CallbackHandler()
        return _langfuse_callback
    except Exception as exc:
        print(f"WARNING: Could not initialize Langfuse callback: {exc}")
        return None


def get_llm(temperature=0.1):
    """Get a Bedrock LLM instance for agent use with Langfuse tracing.

    Automatically configures Langfuse tracing via CallbackHandler if credentials
    are set in environment variables.
    """
    callbacks = []
    langfuse_cb = get_langfuse_callback()
    if langfuse_cb:
        callbacks = [langfuse_cb]

    return ChatBedrockConverse(
        model="us.amazon.nova-micro-v1:0",
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        temperature=temperature,
        max_tokens=1000,
        callbacks=callbacks,
    )


def flush_traces():
    """Flush all pending traces to Langfuse.

    Call this before process exit to ensure traces are sent. In Langfuse v3+
    flushing lives on the client, not the LangChain callback handler.
    """
    if get_langfuse_callback() is None:
        return

    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:
        print(f"WARNING: Could not flush Langfuse traces: {exc}")
