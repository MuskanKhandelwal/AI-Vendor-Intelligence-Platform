# Langfuse Observability Setup

This document describes how Langfuse tracing is configured for the AI Vendor Intelligence Platform.

## Architecture

The project has **two layers of Langfuse tracing**:

### 1. Collector Layer (Data Ingestion)
- **Files**: `collector/langfuse_helper.py`
- **Purpose**: Trace signal classification and data collection operations
- **What gets traced**: 
  - News signal classification (headline → signal type + importance)
  - EDGAR filing processing
  - Collector run metadata
- **Traces stored in**: Supabase `ai_company_signals.langfuse_trace_id` column

### 2. Agent Layer (Analysis & Brief Generation)
- **Files**: `agents/llm.py`, `agents/supervisor.py`
- **Purpose**: Trace LLM calls and agent orchestration
- **What gets traced**:
  - Supervisor routing decisions
  - Financial, Technology, News, Personnel, Competitive agent calls
  - Each LLM invocation (model, tokens, latency)
- **View in**: Langfuse Dashboard

## Configuration

### Environment Variables

Required in `.env`:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...          # From Langfuse Settings > API Keys
LANGFUSE_SECRET_KEY=sk-lf-...          # From Langfuse Settings > API Keys
LANGFUSE_HOST=https://cloud.langfuse.com  # Cloud, US cloud, or self-hosted URL
```

**For EU cloud**, use: `https://eu.cloud.langfuse.com`
**For US cloud**, use: `https://us.cloud.langfuse.com`
**For self-hosted**, use: `https://your-langfuse-instance.com`

### Getting API Keys

1. Log in to https://cloud.langfuse.com (or your instance)
2. Click **Settings** → **API Keys**
3. Copy **Public Key** and **Secret Key**
4. Add to `.env` file (never paste into chat for security)

## Implementation Details

### Agent Layer (LangChain Integration)

Uses LangChain's native **Langfuse CallbackHandler**:

```python
from langfuse.callback import CallbackHandler
from langchain_aws import ChatBedrockConverse

# Handler automatically captures:
callback = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

llm = ChatBedrockConverse(
    model="us.amazon.nova-micro-v1:0",
    callbacks=[callback],  # Automatic tracing
)
```

**Automatically Captured** (no extra code needed):
- Model name: `us.amazon.nova-micro-v1:0`
- Input/output tokens (counted automatically)
- Latency
- Errors and exceptions
- LangChain chain/agent structure

### Collector Layer (Manual Instrumentation)

Uses `collector/langfuse_helper.py` for explicit trace creation:

```python
from collector.langfuse_helper import trace_signal_classification

trace_id = trace_signal_classification(
    company_name="Cohere",
    headline="Cohere releases new API",
    signal_type="product_launch",
    importance_score=85,
    model_name="gpt-4"
)
# trace_id stored in database for cross-referencing
```

## Best Practices

### 1. Always Flush Traces Before Exit

**Scripts** (`agents/run_brief.py`, `collector/run_all.py`):
```python
from agents.llm import flush_traces

try:
    # ... main logic ...
    result = generate_brief("Cohere")
finally:
    flush_traces()  # Always flush before process exit
```

**FastAPI** (automatic):
- `api/main.py` includes CORS middleware
- Traces flush on request completion

### 2. Trace Names Matter

Use descriptive trace names (shown in Langfuse UI):
- ❌ Bad: `trace-1`, `call`, `agent`
- ✅ Good: `supervisor-routing`, `financial-analysis`, `news-sentiment`

The LLM itself determines trace names in our case:
- Supervisor: `supervisor-routing`
- Financial Agent: `financial-analysis`
- Technology Agent: `technology-momentum`
- etc.

### 3. Don't Log Sensitive Data

The default configuration masks:
- Raw database queries (only logged if explicitly added)
- Internal metadata (not included in trace input)
- API keys and credentials (not captured in traced code)

If adding custom fields, avoid:
- User email addresses (if not necessary)
- Database passwords
- API keys
- Confidential competitive data

### 4. Monitor Token Usage

Langfuse automatically calculates costs:
- **Input tokens**: Tracked per call
- **Output tokens**: Tracked per call
- **Cost**: Calculated using AWS Bedrock pricing

View in Langfuse Dashboard:
- **Traces**: Individual calls
- **Analytics**: Cost trends over time
- **Dashboard**: Cost breakdown by agent

### 5. Use Sessions for Multi-Turn Interactions

Not currently implemented but recommended for future:

```python
trace.session_id = "conversation-123"  # Group related traces
```

This would show full conversation flows in Langfuse.

## Troubleshooting

### Traces Not Appearing

1. **Check credentials**: `python collector/langfuse_helper.py`
   - Should see: `Langfuse connection OK. Test trace ID: ...`
   - If not, verify `.env` has correct keys

2. **Check flush is called**: 
   - Scripts must call `flush_traces()`
   - Without it, traces stay in memory and never send

3. **Check host URL**:
   - Cloud: `https://cloud.langfuse.com`
   - US: `https://us.cloud.langfuse.com`
   - EU: `https://eu.cloud.langfuse.com`

4. **Network connectivity**:
   ```bash
   curl -I https://cloud.langfuse.com
   ```

### Wrong Credentials Error

```
WARNING: Langfuse client initialised but trace submission failed.
```

**Fix**:
1. Get fresh keys from Langfuse UI (Settings → API Keys)
2. Update `.env`
3. Restart the application

### Missing Token Counts

If tokens show as `None` in dashboard:
- Bedrock may not return token usage for all model types
- This is normal and doesn't affect other trace data
- Cost calculation uses estimated tokens

## Accessing Traces

### Collector Traces (Past Runs)

Supabase dashboard:
```sql
SELECT company_name, signal_type, langfuse_trace_id
FROM ai_company_signals
WHERE langfuse_trace_id IS NOT NULL
LIMIT 10;
```

Then view in Langfuse:
```
https://cloud.langfuse.com/traces/<trace_id>
```

### Agent Traces (Live)

1. Run brief generation:
   ```bash
   python agents/run_brief.py "Cohere"
   ```

2. Go to Langfuse Dashboard: https://cloud.langfuse.com/traces

3. Filter by:
   - Time range (last 1 hour)
   - Model: `us.amazon.nova-micro-v1:0`
   - Status (success/error)

4. Click a trace to see:
   - Full input/output
   - Agent routing decisions
   - Token usage
   - Latency breakdown

## Files Modified for Langfuse

- `agents/llm.py` - LangChain callback initialization
- `agents/supervisor.py` - Trace flushing on completion
- `collector/langfuse_helper.py` - SDK API update for latest version
- `.env` - Langfuse credentials (not committed)

## References

- **Langfuse Docs**: https://langfuse.com/docs
- **LangChain Integration**: https://langfuse.com/integrations/frameworks/langchain
- **Tracing Best Practices**: https://langfuse.com/docs/tracing
- **Scores & Evaluation**: https://langfuse.com/docs/scores

## Next Steps

Optional enhancements:
1. Add `user_id` or `company_id` tags for multi-tenant filtering
2. Implement `session_id` for conversation grouping
3. Add custom scores (quality rating, factual accuracy)
4. Set up CI/CD experiment gates
5. Build automated quality dashboards
