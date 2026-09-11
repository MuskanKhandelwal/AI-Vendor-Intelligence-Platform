"""Scrapes and normalises news articles and press releases for tracked AI vendors."""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import feedparser
from dotenv import load_dotenv
from groq import BadRequestError, Groq

sys.path.insert(0, os.path.dirname(__file__))
import db
import langfuse_helper
from entity_filter import check_entity, news_query

load_dotenv()

SEED_FILE = Path(__file__).parent / "seed_companies.json"
RATE_LIMIT_SLEEP = 1  # seconds between companies
GROQ_MODEL = "openai/gpt-oss-20b"

CLASSIFICATION_PROMPT = """You are screening news for a procurement intelligence
platform that tracks {company_name}, an AI/machine-learning technology vendor.

Headline: '{title}'

STEP 1a — Identify the grammatical role of "{company_name}" in the headline.
Quote the exact phrase containing the name, then classify that phrase as one of:
- "proper_noun_entity": the name refers to a specific organization acting or
  being acted upon (e.g. "Scale AI raised funding", "Scale AI's CEO departed",
  "Cohere raises $500M").
- "common_word_or_verb": the name is used as an ordinary word/verb, not as an
  organization. Test: could you substitute a synonym (e.g. "grow"/"expand" for
  "scale") and have the sentence still make identical sense? If yes, this is
  the verb, not the company (e.g. "to scale AI inference", "scale AI without
  increasing risk", "before you scale AI").
- "different_organization": a different, specifically-named entity is the
  actual subject/actor of the headline, and {company_name} appears only
  incidentally or as part of a longer descriptive phrase (e.g. "Elevate
  Education Bags Rs 170 Cr To Scale AI-Led Learning Platform" — the actor is
  Elevate Education, not Scale AI).

STEP 1b — Decide "about_company". This matters more than the category.
- Company names are not unique. Unrelated organisations share names with the AI
  vendors we track (e.g. "Cohere Health" is a health-insurance company and
  "Cohere Technologies" is a wireless-RF company; neither is the AI vendor
  Cohere).
- Signals that the headline is a namesake, not the AI vendor: subject matter far
  outside AI/software — telecom RF and radio hardware, defence radar, health
  insurance and clinical operations, aviation, fashion, consumer packaged goods.
- Set "about_company": true ONLY IF step 1a == "proper_noun_entity" AND that
  entity plausibly is the AI/ML vendor {company_name} (not an unrelated
  namesake organisation with the same name).
- Set "about_company": false if step 1a == "common_word_or_verb" or
  "different_organization", or if genuinely uncertain — a wrong-company signal
  becomes a false claim in a procurement brief.
- If "about_company" is false: set "signal_type": "other", "importance_score": 0,
  and "one_line_summary" to a short neutral note on what the headline is
  actually about (the real subject, or that the name was used as a common
  word/verb).

STEP 2 — Choose signal_type (only if about_company is true):
- "negative": ONLY genuine business/financial distress — layoffs, outages,
  bankruptcy, insolvency, product recalls, lawsuits.
- "reputational": controversy, criticism, geopolitical friction, competitive
  pressure, or public-image news that is NOT financial distress. If unsure
  between negative and reputational, choose reputational.
- Otherwise: funding, executive_change, product_launch, partnership,
  regulatory, or other.

STEP 3 — Score importance_score (only if about_company is true):
- funding: 70-95, negative: 65-95, executive_change: 60-85,
  product_launch: 50-80, regulatory: 50-80, partnership: 45-70,
  reputational: 35-65, other: 10-30

STEP 4 — Write one_line_summary: a neutral, factual restatement of the
headline under 150 characters, no editorializing.

EXAMPLES

Headline: "Cohere raises $500M Series D led by Nvidia"
{{"quoted_evidence": "Cohere raises $500M Series D", "grammatical_role": "proper_noun_entity", "about_company": true, "signal_type": "funding", "importance_score": 92, "one_line_summary": "Cohere raised a $500M Series D led by Nvidia."}}

Headline: "Cohere Health names new Chief Medical Officer"
{{"quoted_evidence": "Cohere Health names new Chief Medical Officer", "grammatical_role": "proper_noun_entity", "about_company": false, "signal_type": "other", "importance_score": 0, "one_line_summary": "About Cohere Health, a health-insurance company, not the AI vendor Cohere."}}

Headline: "Cohere faces backlash over data retention policy in EU"
{{"quoted_evidence": "Cohere faces backlash", "grammatical_role": "proper_noun_entity", "about_company": true, "signal_type": "reputational", "importance_score": 50, "one_line_summary": "Cohere criticized in EU over its data retention policy."}}

Headline: "SambaNova Raised $1 Billion To Scale AI Inference"
{{"quoted_evidence": "To Scale AI Inference", "grammatical_role": "common_word_or_verb", "about_company": false, "signal_type": "other", "importance_score": 0, "one_line_summary": "About SambaNova; 'scale' used as a verb, not the company Scale AI."}}

Headline: "Build The Human Foundations Before You Scale AI"
{{"quoted_evidence": "Before You Scale AI", "grammatical_role": "common_word_or_verb", "about_company": false, "signal_type": "other", "importance_score": 0, "one_line_summary": "General advice article; 'scale AI' used as a verb phrase, not the company."}}

Headline: "How defense teams can scale AI without increasing risk"
{{"quoted_evidence": "scale AI without increasing risk", "grammatical_role": "common_word_or_verb", "about_company": false, "signal_type": "other", "importance_score": 0, "one_line_summary": "General advice on AI adoption; 'scale' used as a verb, not the company."}}

Headline: "Elevate Education Bags Rs 170 Cr To Scale AI-Led Learning Platform"
{{"quoted_evidence": "Elevate Education Bags Rs 170 Cr", "grammatical_role": "different_organization", "about_company": false, "signal_type": "other", "importance_score": 0, "one_line_summary": "Elevate Education raised funding; 'Scale AI-Led' is descriptive, not the company Scale AI."}}

Return ONLY the JSON object below — no markdown fences, no preamble, no
trailing text:
{{
  "quoted_evidence": string,
  "grammatical_role": one of [proper_noun_entity, common_word_or_verb, different_organization],
  "about_company": true or false,
  "signal_type": one of [funding, executive_change, product_launch, partnership, negative, reputational, regulatory, other],
  "importance_score": integer 0-100,
  "one_line_summary": string under 150 chars
}}
"""

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

def _unverified_default(title: str) -> dict:
    """Classification used when the entity check could not be completed.

    Fails CLOSED, matching what CLASSIFICATION_PROMPT already tells the model:
    an uncertain call must be false, because a wrong-company signal becomes a
    false claim in a procurement brief, whereas a dropped signal is a visible
    gap that confidence scoring already flags.

    This previously defaulted to about_company=True, which meant any API or
    JSON failure silently waved the headline through with the entity check
    never having run.
    """
    return {
        "about_company": False,
        "signal_type": "other",
        "importance_score": 0,
        "one_line_summary": f"Entity check unavailable, not stored: {title[:105]}",
    }


def classify_article(
    groq: Groq, company_name: str, title: str
) -> tuple[dict, bool, str | None]:
    """Call Groq to classify a news article.

    Returns (classification, hit_rate_limit, langfuse_trace_id). The trace is
    created here rather than by the caller so that it records the real prompt
    and the real response, and so that it is only created when an LLM call
    actually happened -- rule-classified headlines get None.
    """
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
                # v3 emits quoted_evidence + grammatical_role before the verdict.
                # Measured peak is 329 tokens; at 200 roughly a quarter of calls
                # died with json_validate_failed and fell through to the default.
                max_tokens=500,
                temperature=0,
                # gpt-oss models emit reasoning tokens before the answer. "low"
                # roughly halves them; the classification is unchanged.
                reasoning_effort="low",
            )
            json_text = response.choices[0].message.content.strip()
            classification = json.loads(json_text)
            trace_id = langfuse_helper.trace_collector_call(
                trace_name="news-classification",
                company_name=company_name,
                prompt=prompt,
                response_text=json_text,
                model_name=GROQ_MODEL,
                response=response,
                headline=title,
                signal_type=classification.get("signal_type"),
                importance_score=classification.get("importance_score"),
                about_company=classification.get("about_company"),
                grammatical_role=classification.get("grammatical_role"),
            )
            return classification, False, trace_id
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
                    return _unverified_default(title), True, None
            # JSON-mode validation failure: the model ran out of tokens before
            # closing the object, or broke the schema. Distinct from a transient
            # error - it points at max_tokens or the prompt, so say so.
            elif isinstance(exc, BadRequestError) or "json_validate_failed" in error_message:
                print(
                    f"    [json validate failed] response truncated or off-schema "
                    f"(check max_tokens): {title[:60]}"
                )
                return _unverified_default(title), False, None
            # Non-rate-limit errors: return defaults immediately
            elif isinstance(exc, (json.JSONDecodeError, KeyError, AttributeError)):
                return _unverified_default(title), False, None
            else:
                # Other exceptions: return defaults
                return _unverified_default(title), False, None


# ---------------------------------------------------------------------------
# Per-company processing
# ---------------------------------------------------------------------------

def _process_company(company: dict, groq: Groq) -> tuple[int, int, bool, int]:
    """Fetch and classify news for a company.

    Returns (signals_added, errors, hit_rate_limit, rejected), where `rejected`
    counts headlines dropped because they are about a different organisation.
    If hit_rate_limit is True, the caller should stop processing and retry later.
    """
    name = company["name"]
    ticker = company.get("ticker")
    signals_added = 0
    errors = 0
    rejected = 0
    hit_rate_limit = False

    # Fetch Google News RSS. The query excludes known namesakes upstream so
    # colliding organisations never enter the feed in the first place.
    query = quote_plus(news_query(company))
    url = (
        f"https://news.google.com/rss/search"
        f"?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            return 0, 0, False, 0
    except Exception as exc:
        print(f"  News feed fetch failed for {name}: {exc}")
        return 0, 1, False, 0

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

        # Deterministic entity check before anything else: Google News matches
        # on the name alone, so namesake organisations still slip through.
        verdict = check_entity(company, title, entry.get("summary", ""))
        if not verdict.ok:
            rejected += 1
            print(f"    [skip: {verdict.reason}] {title[:70]}")
            continue

        # Try rule-based classification first (saves tokens)
        classification = classify_article_rules(title)
        # Rule-matched headlines never reach the model, so they have no trace.
        # This used to emit a Langfuse "generation" for them anyway, recording
        # a model call that never happened.
        trace_id = None
        if classification is None:
            # No confident rule match — fall through to Groq
            classification, hit_limit, trace_id = classify_article(
                groq, name, entry.get("title", "")
            )
            if hit_limit:
                # Rate limit hit — stop processing this company
                return signals_added, errors, True, rejected

        # The LLM is the second line of defence for namesakes the seed config
        # doesn't know about. Rule-matched headlines skip it and default to True.
        if classification.get("about_company", True) is False:
            rejected += 1
            print(f"    [skip: classifier says not this company] {title[:70]}")
            continue

        signal_type = classification.get("signal_type", "other")
        importance_score = classification.get("importance_score", 30)
        one_line_summary = classification.get("one_line_summary", title[:150])
        # The model's own reasoning for the about_company verdict. Kept because
        # it is the only record of WHY a headline was accepted: the Langfuse
        # trace stores a reconstructed input/output, not the real exchange.
        # Absent on rule-classified headlines, which never reach the LLM.
        grammatical_role = classification.get("grammatical_role")
        quoted_evidence = classification.get("quoted_evidence")

        funding_amount_millions = (
            extract_funding_amount(title) if signal_type == "funding" else None
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
                    "grammatical_role": grammatical_role,
                    "quoted_evidence": quoted_evidence,
                },
                langfuse_trace_id=trace_id,
            )
            if inserted:
                signals_added += 1
        except Exception as exc:
            print(f"    [insert error] {title}: {exc}")
            errors += 1

    return signals_added, errors, hit_rate_limit, rejected


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
    total_rejected = 0
    started_at = datetime.now(tz=timezone.utc)
    companies_processed = 0

    print(f"News collector starting — {total} companies to process.\n")

    for i, company in enumerate(all_companies, start=1):
        name = company["name"]
        print(f"Processing {name} ({i}/{total})...")

        try:
            signals, errors, hit_rate_limit, rejected = _process_company(company, groq)
            total_signals += signals
            total_errors += errors
            total_rejected += rejected
            companies_processed += 1
            print(
                f"  -> {signals} signal(s) added, {rejected} rejected "
                f"(wrong entity), {errors} error(s)."
            )

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

    print(
        f"\nDone. {total_signals} signals added, {total_rejected} rejected as "
        f"wrong-entity, {total_errors} errors. "
        f"({companies_processed}/{total} companies processed.)"
    )


if __name__ == "__main__":
    run()
