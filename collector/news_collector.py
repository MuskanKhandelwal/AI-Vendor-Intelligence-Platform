"""Scrapes and normalises news articles and press releases for tracked AI vendors."""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import feedparser
from dotenv import load_dotenv
from groq import Groq

sys.path.insert(0, os.path.dirname(__file__))
import db
import langfuse_helper

load_dotenv()

SEED_FILE = Path(__file__).parent / "seed_companies.json"
RATE_LIMIT_SLEEP = 1  # seconds between companies
GROQ_MODEL = "llama-3.3-70b-versatile"

CLASSIFICATION_PROMPT = """Classify this news headline about the AI company {company_name}.
Headline: '{title}'

Return only valid JSON with these exact fields:
{{
  "signal_type": one of [funding, executive_change, product_launch, partnership, negative, regulatory, other],
  "importance_score": integer 0-100,
  "one_line_summary": string under 150 chars
}}

Scoring guide:
- funding rounds: 85
- executive changes: 75
- product launches: 70
- partnerships: 60
- negative news (layoffs, outages, lawsuits): 80
- regulatory: 65
- other: 30"""

FUNDING_AMOUNT_RE = re.compile(r'\$\s?(\d+(?:\.\d+)?)\s*(million|billion|[MB])\b', re.I)


def extract_funding_amount(title: str) -> float | None:
    """Extract a dollar amount from a headline, normalized to USD millions.

    e.g. "raises $200M" -> 200.0, "raises $1.2 billion" -> 1200.0
    """
    match = FUNDING_AMOUNT_RE.search(title)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("b"):
        amount *= 1000
    return amount


# ---------------------------------------------------------------------------
# Rule-based pre-classification
# ---------------------------------------------------------------------------

def classify_article_rules(title: str) -> dict | None:
    """Classify headline with keyword rules. Returns dict if confident, None to fall through to LLM."""
    title_lower = title.lower()

    # Funding/capital
    if any(kw in title_lower for kw in ["series a", "series b", "series c", "series d", "series e",
                                         "funding", "raised", "investment", "venture", "capital"]):
        return {"signal_type": "funding", "importance_score": 85, "one_line_summary": title[:150]}

    # Layoffs/negative
    if any(kw in title_lower for kw in ["layoffs", "layoff", "lay off", "cutting staff", "cuts jobs",
                                         "bankruptcy", "bankrupt", "collapse", "shutdown", "closes"]):
        return {"signal_type": "negative", "importance_score": 80, "one_line_summary": title[:150]}

    # Partnerships
    if any(kw in title_lower for kw in ["partnership", "partner", "collaborate", "collaboration",
                                         "integrates with", "integration", "teams up", "allied"]):
        return {"signal_type": "partnership", "importance_score": 60, "one_line_summary": title[:150]}

    # Product launches/releases
    if any(kw in title_lower for kw in ["launches", "release", "release", "new product", "introduces",
                                         "unveils", "announces new", "debut", "launches new"]):
        return {"signal_type": "product_launch", "importance_score": 70, "one_line_summary": title[:150]}

    # Executive changes
    if any(kw in title_lower for kw in ["ceo", "cto", "founder", "executive", "resign", "resigns",
                                         "appoints", "appointed", "joins", "departs", "departure",
                                         "leaves", "left company"]):
        return {"signal_type": "executive_change", "importance_score": 75, "one_line_summary": title[:150]}

    # Regulatory/legal
    if any(kw in title_lower for kw in ["lawsuit", "sued", "settlement", "regulation", "regulatory",
                                         "legal", "court", "fined", "fine", "compliance", "investigation"]):
        return {"signal_type": "regulatory", "importance_score": 65, "one_line_summary": title[:150]}

    # No confident match — fall through to LLM
    return None


# ---------------------------------------------------------------------------
# Groq classification
# ---------------------------------------------------------------------------

def classify_article(groq: Groq, company_name: str, title: str) -> tuple[dict, bool]:
    """Call Groq to classify a news article. Returns (dict, hit_rate_limit) tuple."""
    prompt = CLASSIFICATION_PROMPT.format(company_name=company_name, title=title)
    attempts = 0
    max_attempts = 2

    while attempts < max_attempts:
        attempts += 1
        try:
            response = groq.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=200,
                temperature=0,
            )
            json_text = response.choices[0].message.content.strip()
            classification = json.loads(json_text)
            return classification, False
        except Exception as exc:
            error_message = str(exc)
            # Check for 429 rate limit error
            if "429" in error_message or "rate_limit" in error_message.lower():
                if attempts < max_attempts:
                    # Parse wait time from error message
                    match = re.search(r'try again in (\d+)m([\d.]+)s', error_message)
                    if match:
                        wait = int(match.group(1)) * 60 + float(match.group(2)) + 5
                    else:
                        wait = 180  # default 3 minutes
                    print(f"  [rate limit] Daily token limit reached. Waiting {wait:.0f}s before retrying...")
                    time.sleep(wait)
                    continue
                else:
                    # Second attempt failed, signal rate limit and return defaults
                    return (
                        {
                            "signal_type": "other",
                            "importance_score": 30,
                            "one_line_summary": title[:150],
                        },
                        True,  # hit_rate_limit = True
                    )
            # Non-rate-limit errors: return defaults immediately
            elif isinstance(exc, (json.JSONDecodeError, KeyError, AttributeError)):
                return (
                    {
                        "signal_type": "other",
                        "importance_score": 30,
                        "one_line_summary": title[:150],
                    },
                    False,
                )
            else:
                # Other exceptions: return defaults
                return (
                    {
                        "signal_type": "other",
                        "importance_score": 30,
                        "one_line_summary": title[:150],
                    },
                    False,
                )


# ---------------------------------------------------------------------------
# Per-company processing
# ---------------------------------------------------------------------------

def _process_company(company: dict, groq: Groq) -> tuple[int, int, bool]:
    """Fetch and classify news for a company. Returns (signals_added, errors, hit_rate_limit).

    If hit_rate_limit is True, the caller should stop processing and retry later.
    """
    name = company["name"]
    ticker = company.get("ticker")
    signals_added = 0
    errors = 0
    hit_rate_limit = False

    # Fetch Google News RSS
    url = (
        f"https://news.google.com/rss/search"
        f"?q={name.replace(' ', '+')}+AI&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return 0, 0, False
    except Exception as exc:
        print(f"  News feed fetch failed for {name}: {exc}")
        return 0, 1, False

    now = datetime.now(tz=timezone.utc)

    for entry in feed.entries:
        try:
            pub_date = parsedate_to_datetime(entry.published)
        except (TypeError, AttributeError, ValueError):
            continue

        # Filter to last 7 days
        days_old = (now - pub_date).days
        if days_old > 7:
            continue

        title = entry.get("title", "Untitled")[:300]
        link = entry.get("link", "")

        # Try rule-based classification first (saves tokens)
        classification = classify_article_rules(title)
        if classification is None:
            # No confident rule match — fall through to Groq
            classification, hit_limit = classify_article(groq, name, entry.get("title", ""))
            if hit_limit:
                # Rate limit hit — stop processing this company
                return signals_added, errors, True

        signal_type = classification.get("signal_type", "other")
        importance_score = classification.get("importance_score", 30)
        one_line_summary = classification.get("one_line_summary", title[:150])

        funding_amount_millions = (
            extract_funding_amount(title) if signal_type == "funding" else None
        )

        # Log to Langfuse
        trace_id = langfuse_helper.trace_signal_classification(
            company_name=name,
            headline=entry.get("title", ""),
            signal_type=signal_type,
            importance_score=importance_score,
            model_name=GROQ_MODEL,
        )

        # Insert signal
        try:
            inserted = db.insert_signal(
                company_name=name,
                ticker=ticker,
                signal_type=signal_type,
                signal_date=pub_date.date(),
                headline=title,
                summary=one_line_summary,
                source_url=link,
                importance_score=importance_score,
                raw_data={
                    "feed_title": feed.feed.get("title", ""),
                    "original_title": entry.get("title", ""),
                    "funding_amount_millions": funding_amount_millions,
                },
                langfuse_trace_id=trace_id,
            )
            if inserted:
                signals_added += 1
        except Exception as exc:
            print(f"    [insert error] {title}: {exc}")
            errors += 1

    return signals_added, errors, hit_rate_limit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run() -> None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set in .env. Cannot classify articles."
        )

    groq = Groq(api_key=api_key)

    with open(SEED_FILE) as f:
        all_companies = json.load(f)

    total = len(all_companies)
    total_signals = 0
    total_errors = 0
    started_at = datetime.now(tz=timezone.utc)
    companies_processed = 0

    print(f"News collector starting — {total} companies to process.\n")

    for i, company in enumerate(all_companies, start=1):
        name = company["name"]
        print(f"Processing {name} ({i}/{total})...")

        try:
            signals, errors, hit_rate_limit = _process_company(company, groq)
            total_signals += signals
            total_errors += errors
            companies_processed += 1
            print(f"  -> {signals} signal(s) added, {errors} error(s).")

            if hit_rate_limit:
                print(f"  [RATE LIMIT] Hit Groq daily limit at {name}. Stopping collection.")
                print(f"  Resume tomorrow or check token usage in Langfuse.")
                break

        except Exception as exc:
            print(f"  ERROR processing {name}: {exc}")
            total_errors += 1
            companies_processed += 1

        if i < total:
            time.sleep(RATE_LIMIT_SLEEP)

    db.log_run(
        collector_name="news_collector",
        companies_processed=companies_processed,
        signals_added=total_signals,
        errors=total_errors,
        started_at=started_at,
    )

    print(f"\nDone. {total_signals} signals added, {total_errors} errors. ({companies_processed}/{total} companies processed.)")


if __name__ == "__main__":
    run()
