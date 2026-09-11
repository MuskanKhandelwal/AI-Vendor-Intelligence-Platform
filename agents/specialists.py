"""Five specialist agents for intelligence analysis."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain.agents import create_agent
from agents.llm import get_llm
from agents.tools import (
    compute_financial_health_score,
    query_funding_signals,
    query_executive_changes,
    query_github_signals,
    query_arxiv_signals,
    query_news_signals,
    query_negative_signals,
    query_annual_filings,
    get_signal_summary,
    query_neo4j_competitors,
    query_neo4j_investors,
    query_neo4j_company_info,
    query_all_signals_count,
)


def _sum_usage(messages: list) -> tuple[int, int]:
    """Sum input/output tokens across all AIMessages in a ReAct agent run.

    LangChain's ChatBedrockConverse attaches real usage_metadata (input_tokens,
    output_tokens) to each AIMessage, including intermediate tool-calling turns.
    """
    input_tokens = 0
    output_tokens = 0
    for message in messages:
        usage = getattr(message, "usage_metadata", None)
        if usage:
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)
    return input_tokens, output_tokens


# ---------------------------------------------------------------------------
# Financial Agent
# ---------------------------------------------------------------------------

FINANCIAL_PROMPT = """You are a financial analyst evaluating AI vendors for \
enterprise procurement decisions. Given a company name:
1. Call compute_financial_health_score FIRST — it returns the exact, \
deterministic FINANCIAL HEALTH SCORE and reasoning you must report. Never \
invent your own score. If it says data is unavailable, report the score as \
UNKNOWN rather than guessing a number.
2. Use your other tools to gather supporting detail, then provide:
   FINANCIAL HEALTH SCORE: [the number from compute_financial_health_score, or UNKNOWN]
   KEY FINDINGS: 3 bullet points with dates
   RISK FLAGS: any concerning signals
   SOURCES: cite signal dates and headlines
Be concise. If no data found, say so honestly."""

financial_agent = create_agent(
    model=get_llm(),
    tools=[
        compute_financial_health_score,
        query_funding_signals,
        query_annual_filings,
        query_executive_changes,
        get_signal_summary,
    ],
    system_prompt=FINANCIAL_PROMPT,
)


def run_financial_agent(company_name: str) -> tuple[str, int, int]:
    """Run financial agent on a company. Returns (output, input_tokens, output_tokens)."""
    try:
        result = financial_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze {company_name}",
                    }
                ]
            }
        )
        input_tokens, output_tokens = _sum_usage(result["messages"])
        return result["messages"][-1].content, input_tokens, output_tokens
    except Exception as exc:
        return f"Error running financial agent: {exc}", 0, 0


# ---------------------------------------------------------------------------
# Technology Agent
# ---------------------------------------------------------------------------

TECHNOLOGY_PROMPT = """You are a technology analyst evaluating AI vendors. \
Given a company name, use your tools to gather technology signals then provide:
1. TECHNOLOGY MOMENTUM SCORE: 0-100
2. RECENT RELEASES: list with dates
3. RESEARCH OUTPUT: paper count and topics
4. GITHUB ACTIVITY: release velocity assessment
Be concise. If no data found, say so honestly."""

technology_agent = create_agent(
    model=get_llm(),
    tools=[
        query_github_signals,
        query_arxiv_signals,
        get_signal_summary,
        query_all_signals_count,
    ],
    system_prompt=TECHNOLOGY_PROMPT,
)


def run_technology_agent(company_name: str) -> tuple[str, int, int]:
    """Run technology agent on a company. Returns (output, input_tokens, output_tokens)."""
    try:
        result = technology_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze {company_name}",
                    }
                ]
            }
        )
        input_tokens, output_tokens = _sum_usage(result["messages"])
        return result["messages"][-1].content, input_tokens, output_tokens
    except Exception as exc:
        return f"Error running technology agent: {exc}", 0, 0


# ---------------------------------------------------------------------------
# News Agent
# ---------------------------------------------------------------------------

NEWS_PROMPT = """You are a news and sentiment analyst evaluating AI vendors. \
Given a company name, use your tools to gather news signals then provide:
1. SENTIMENT: positive/neutral/negative with reasoning
2. KEY EVENTS: top 3 most important recent events
3. NEGATIVE FLAGS: any concerning news
4. PARTNERSHIPS: notable announcements
Be concise. If no data found, say so honestly."""

news_agent = create_agent(
    model=get_llm(),
    tools=[
        query_news_signals,
        query_negative_signals,
        get_signal_summary,
    ],
    system_prompt=NEWS_PROMPT,
)


def run_news_agent(company_name: str) -> tuple[str, int, int]:
    """Run news agent on a company. Returns (output, input_tokens, output_tokens)."""
    try:
        result = news_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze {company_name}",
                    }
                ]
            }
        )
        input_tokens, output_tokens = _sum_usage(result["messages"])
        return result["messages"][-1].content, input_tokens, output_tokens
    except Exception as exc:
        return f"Error running news agent: {exc}", 0, 0


# ---------------------------------------------------------------------------
# Personnel Agent
# ---------------------------------------------------------------------------

PERSONNEL_PROMPT = """You are a personnel analyst evaluating AI vendor \
leadership stability. Given a company name, use your tools to gather \
personnel signals then provide:
1. PERSONNEL STABILITY SCORE: 0-100
2. EXECUTIVE CHANGES: list any with dates and roles
3. STABILITY ASSESSMENT: one paragraph
Be concise. If no data found, say so honestly."""

personnel_agent = create_agent(
    model=get_llm(),
    tools=[
        query_executive_changes,
        query_annual_filings,
        get_signal_summary,
    ],
    system_prompt=PERSONNEL_PROMPT,
)


def run_personnel_agent(company_name: str) -> tuple[str, int, int]:
    """Run personnel agent on a company. Returns (output, input_tokens, output_tokens)."""
    try:
        result = personnel_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze {company_name}",
                    }
                ]
            }
        )
        input_tokens, output_tokens = _sum_usage(result["messages"])
        return result["messages"][-1].content, input_tokens, output_tokens
    except Exception as exc:
        return f"Error running personnel agent: {exc}", 0, 0


# ---------------------------------------------------------------------------
# Competitive Agent
# ---------------------------------------------------------------------------

COMPETITIVE_PROMPT = """You are a competitive intelligence analyst evaluating \
AI vendors. Given a company name, use your tools to gather competitive signals \
then provide:
1. MARKET POSITION: one paragraph
2. DIRECT COMPETITORS: list with stages
3. KEY INVESTORS: list
4. COMPETITIVE RISKS: any concerning signals
Be concise. If no data found, say so honestly."""

competitive_agent = create_agent(
    model=get_llm(),
    tools=[
        query_neo4j_competitors,
        query_neo4j_investors,
        query_neo4j_company_info,
        query_news_signals,
    ],
    system_prompt=COMPETITIVE_PROMPT,
)


def run_competitive_agent(company_name: str) -> tuple[str, int, int]:
    """Run competitive agent on a company. Returns (output, input_tokens, output_tokens)."""
    try:
        result = competitive_agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Analyze {company_name}",
                    }
                ]
            }
        )
        input_tokens, output_tokens = _sum_usage(result["messages"])
        return result["messages"][-1].content, input_tokens, output_tokens
    except Exception as exc:
        return f"Error running competitive agent: {exc}", 0, 0


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    company = "Cohere"
    print(f"Running all 5 specialist agents for {company}\n")

    print("=" * 70)
    print("FINANCIAL AGENT")
    print("=" * 70)
    output, in_tok, out_tok = run_financial_agent(company)
    print(output)
    print(f"[tokens] input={in_tok} output={out_tok}")
    print()

    print("=" * 70)
    print("TECHNOLOGY AGENT")
    print("=" * 70)
    output, in_tok, out_tok = run_technology_agent(company)
    print(output)
    print(f"[tokens] input={in_tok} output={out_tok}")
    print()

    print("=" * 70)
    print("NEWS AGENT")
    print("=" * 70)
    output, in_tok, out_tok = run_news_agent(company)
    print(output)
    print(f"[tokens] input={in_tok} output={out_tok}")
    print()

    print("=" * 70)
    print("PERSONNEL AGENT")
    print("=" * 70)
    output, in_tok, out_tok = run_personnel_agent(company)
    print(output)
    print(f"[tokens] input={in_tok} output={out_tok}")
    print()

    print("=" * 70)
    print("COMPETITIVE AGENT")
    print("=" * 70)
    output, in_tok, out_tok = run_competitive_agent(company)
    print(output)
    print(f"[tokens] input={in_tok} output={out_tok}")
    print()
