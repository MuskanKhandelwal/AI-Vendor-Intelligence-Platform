"""Deterministic scoring rubrics computed from collected signals.

Replaces LLM-guessed 0-100 scores with reproducible, explainable math based
on what's actually in the database, instead of a paid financial data API.
"""

import sys
import os
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import collector.db as db


def _days_since(signal_date) -> int | None:
    if signal_date is None:
        return None
    return (date.today() - signal_date).days


# Headline keywords for a *financial-distress* negative — the only kind that
# should move the Financial Health score. Reputational/geopolitical/competitive
# bad press (e.g. "China warns of security backdoor") is negative sentiment but
# NOT financial distress, so it must not drag this score down.
_DISTRESS_KEYWORDS = (
    "layoff", "lay off", "cutting staff", "cuts jobs", "job cuts",
    "bankruptcy", "bankrupt", "insolvency", "insolvent", "collapse",
    "shutdown", "shuts down", "shutting down", "wind down", "winding down",
    "restructuring", "missed payroll", "default", "going out of business",
    "closes doors", "ceases operations",
)


def _is_financial_distress(headline: str | None) -> bool:
    """True if a 'negative' signal's headline indicates actual financial
    distress (layoffs, bankruptcy, insolvency) rather than reputational news."""
    if not headline:
        return False
    hl = headline.lower()
    return any(kw in hl for kw in _DISTRESS_KEYWORDS)


def compute_financial_health(company_name: str) -> dict:
    """Compute a deterministic Financial Health score (0-100) with reasoning.

    Signals considered:
      - Funding recency + amount (bigger, more recent rounds score higher)
      - Negative signals (layoffs/bankruptcy) in the last 90 days
      - Executive churn (frequent changes signal instability)
      - SEC annual filings (public company transparency)

    Returns a dict with `score` (None if no data), `confidence`, and
    `reasoning` (list of strings showing exactly how the score was built).
    """
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT signal_type, signal_date, importance_score, raw_data, headline
                FROM ai_company_signals
                WHERE company_name = %s
                  AND signal_type IN ('funding', 'negative', 'executive_change', 'annual_filing')
                ORDER BY signal_date DESC
                """,
                (company_name,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return {
            "score": None,
            "confidence": "none",
            "reasoning": ["No financial signals collected for this company."],
        }

    score = 50
    reasoning = ["Baseline: 50/100 (neutral, no data)"]

    funding_rows = [r for r in rows if r[0] == "funding"]
    negative_rows = [r for r in rows if r[0] == "negative"]
    exec_rows = [r for r in rows if r[0] == "executive_change"]
    filing_rows = [r for r in rows if r[0] == "annual_filing"]

    if funding_rows:
        _, signal_date, _, raw, _ = funding_rows[0]
        days = _days_since(signal_date)
        amount = (raw or {}).get("funding_amount_millions")

        if days is not None and days <= 180:
            score += 25
            reasoning.append(f"+25: Funding signal within last {days} days")
        elif days is not None and days <= 365:
            score += 10
            reasoning.append(f"+10: Funding signal within last year ({days} days ago)")
        else:
            reasoning.append("+0: Most recent funding signal is over a year old")

        if amount:
            if amount >= 100:
                score += 15
                reasoning.append(f"+15: Large round (${amount:.0f}M+)")
            elif amount >= 20:
                score += 8
                reasoning.append(f"+8: Mid-size round (${amount:.0f}M)")
    else:
        reasoning.append("+0: No funding signal on record")

    if negative_rows:
        # Only genuine financial-distress negatives (layoffs, bankruptcy, etc.)
        # affect Financial Health. Reputational bad press does not — it's tracked
        # under News & Sentiment instead. Headline is r[4].
        recent_distress = [
            r for r in negative_rows
            if (_days_since(r[1]) or 9999) <= 90 and _is_financial_distress(r[4])
        ]
        recent_negative_total = [
            r for r in negative_rows if (_days_since(r[1]) or 9999) <= 90
        ]
        if recent_distress:
            # Scale the penalty with the number of distinct distress signals,
            # capped, instead of a flat -30 regardless of count.
            penalty = min(30, 10 * len(recent_distress))
            score -= penalty
            reasoning.append(
                f"-{penalty}: {len(recent_distress)} financial-distress "
                "signal(s) (layoffs/bankruptcy/insolvency) in last 90 days"
            )
        elif recent_negative_total:
            reasoning.append(
                f"+0: {len(recent_negative_total)} recent negative press "
                "signal(s), but none indicate financial distress (excluded "
                "from Financial Health)"
            )

    if len(exec_rows) >= 3:
        score -= 10
        reasoning.append(
            f"-10: High executive churn ({len(exec_rows)} changes on record)"
        )

    if filing_rows:
        score += 5
        reasoning.append("+5: Public company with SEC filings (higher transparency)")

    score = max(0, min(100, score))
    confidence = "high" if len(rows) >= 4 else "low"

    return {
        "score": score,
        "confidence": confidence,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# Per-dimension confidence (deterministic, evidence-based)
# ---------------------------------------------------------------------------
#
# These power the human-fallback mechanism: a brief whose overall confidence
# is low gets flagged for human review. Confidence is computed purely from how
# much evidence we actually collected — NO extra LLM calls — so it adds no
# meaningful latency to brief generation.

# signal_type -> which dimension(s) it informs. A type may inform more than one
# (e.g. an executive change is both a financial-stability and personnel signal).
_DIMENSION_SIGNALS = {
    "financial": {"funding", "negative", "executive_change", "annual_filing"},
    "technology": {"github_release", "github_activity", "research_paper"},
    "news": {
        "product_launch", "partnership", "negative", "reputational",
        "regulatory", "other", "acquisition",
    },
    "personnel": {"executive_change"},
}

# Evidence-count thresholds per dimension: (high_min, low_min).
#   count == 0            -> "none"
#   0 < count < low_min   -> "none"  (too thin to trust)
#   low_min <= count < high_min -> "low"
#   count >= high_min     -> "high"
_DIMENSION_THRESHOLDS = {
    "financial": (4, 1),
    "technology": (4, 1),
    "news": (4, 1),
    "personnel": (3, 1),
    "competitive": (3, 1),
}


def _level_from_count(count: int, high_min: int, low_min: int) -> str:
    if count >= high_min:
        return "high"
    if count >= low_min:
        return "low"
    return "none"


def _signal_stats(company_name: str) -> dict:
    """One grouped query returning signal statistics for a company:

      - counts_by_type: {signal_type: count}
      - total: total signal count
      - negative_count: count of 'negative' signals
      - max_single_day: (date, count) of the largest same-day batch

    The per-day breakdown lets us detect quality anomalies (e.g. a huge batch
    of signals all dated the same day, which usually means a noisy collection
    run rather than 100 genuinely distinct events).
    """
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT signal_type, signal_date, COUNT(*)
                FROM ai_company_signals
                WHERE company_name = %s
                GROUP BY signal_type, signal_date
                """,
                (company_name,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    counts_by_type: dict[str, int] = {}
    per_day: dict = {}
    negative_count = 0
    total = 0
    for sig_type, sig_date, count in rows:
        counts_by_type[sig_type] = counts_by_type.get(sig_type, 0) + count
        per_day[sig_date] = per_day.get(sig_date, 0) + count
        total += count
        if sig_type == "negative":
            negative_count += count

    max_single_day = max(per_day.items(), key=lambda kv: kv[1], default=(None, 0))
    return {
        "counts_by_type": counts_by_type,
        "total": total,
        "negative_count": negative_count,
        "max_single_day": max_single_day,
    }


def _competitor_count(company_name: str) -> int | None:
    """Count competitors in the Neo4j graph.

    Returns the count on success (0 is a valid answer — genuinely no
    competitors on record), or ``None`` if the graph is unavailable
    (missing config, unreachable, driver not installed). ``None`` lets the
    caller treat an infra outage as "unknown" rather than falsely flagging
    every brief as low-confidence."""
    try:
        from agents.tools import _get_neo4j_driver

        driver = _get_neo4j_driver()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (c:Company {name: $name})-[:COMPETES_WITH]-(:Company)
                RETURN count(*) AS n
                """,
                name=company_name,
            )
            record = result.single()
        driver.close()
        return int(record["n"]) if record else 0
    except Exception:
        return None


def compute_confidence(company_name: str) -> dict:
    """Deterministic per-dimension confidence for a company's brief.

    Returns a dict with:
      - `dimensions`: {dimension: {"level": high|low|none, "evidence_count": int}}
      - `overall`: high|low|none
      - `needs_review`: bool  (True when a human should look before this ships)

    Purely count-based — no LLM calls — so it is fast and reproducible.
    """
    stats = _signal_stats(company_name)
    counts = stats["counts_by_type"]

    dimensions: dict[str, dict] = {}
    for dim, sig_types in _DIMENSION_SIGNALS.items():
        evidence = sum(counts.get(t, 0) for t in sig_types)
        high_min, low_min = _DIMENSION_THRESHOLDS[dim]
        dimensions[dim] = {
            "level": _level_from_count(evidence, high_min, low_min),
            "evidence_count": evidence,
        }

    # Competitive lives in the graph, not the signals table.
    comp_count = _competitor_count(company_name)
    high_min, low_min = _DIMENSION_THRESHOLDS["competitive"]
    if comp_count is None:
        # Graph unavailable — mark unknown and exclude from the review tally so
        # an infra outage doesn't falsely flag every brief.
        dimensions["competitive"] = {"level": "unknown", "evidence_count": None}
    else:
        dimensions["competitive"] = {
            "level": _level_from_count(comp_count, high_min, low_min),
            "evidence_count": comp_count,
        }

    # Roll up to an overall verdict. Bias toward caution: any missing dimension
    # or several thin ones should pull a human in. "unknown" dimensions are
    # excluded — we don't punish a company for our own graph being down.
    levels = [d["level"] for d in dimensions.values() if d["level"] != "unknown"]
    none_count = levels.count("none")
    low_count = levels.count("low")

    if none_count >= 3:
        overall = "none"
    elif none_count >= 1 or low_count >= 2:
        overall = "low"
    else:
        overall = "high"

    # Quality anomalies — these force a human review even when volume is high,
    # because volume alone can hide misclassified / noisy data. This is the
    # gap that let a wrong high-volume brief (e.g. 63 mislabeled "negatives")
    # slip through a purely count-based gate.
    anomalies: list[str] = []
    total = stats["total"]
    if total >= 20:
        neg_share = stats["negative_count"] / total
        if neg_share >= 0.30:
            anomalies.append(
                f"negative signals are {neg_share:.0%} of all data "
                f"({stats['negative_count']}/{total}) — possible "
                "misclassification of reputational news as negative"
            )
        batch_date, batch_count = stats["max_single_day"]
        if batch_count / total >= 0.40 and batch_count >= 15:
            anomalies.append(
                f"{batch_count}/{total} signals share one date ({batch_date}) "
                "— likely a noisy single collection run, not distinct events"
            )

    needs_review = overall != "high" or bool(anomalies)

    return {
        "dimensions": dimensions,
        "overall": overall,
        "anomalies": anomalies,
        "needs_review": needs_review,
    }


# ---------------------------------------------------------------------------
# Evidence — the actual, dated, sourced signals behind a brief
# ---------------------------------------------------------------------------
#
# This is what makes the product more than a generic chatbot: every claim can
# be traced to a specific collected signal with a source link. The UI surfaces
# these under each brief section so procurement can verify the receipts.

# Each signal_type shows under exactly ONE dimension in the evidence UI, so a
# signal never appears twice.
_DISPLAY_DIMENSION = {
    "funding": "financial",
    "annual_filing": "financial",
    "github_release": "technology",
    "github_activity": "technology",
    "research_paper": "technology",
    "executive_change": "personnel",
    "product_launch": "news",
    "partnership": "news",
    "negative": "news",
    "reputational": "news",
    "regulatory": "news",
    "acquisition": "news",
    "other": "news",
}

_EVIDENCE_DIMENSIONS = ("financial", "technology", "news", "personnel")


def get_evidence(company_name: str, per_dimension: int = 6) -> dict:
    """Return the top supporting signals for a company, grouped by dimension.

    Each item has headline, signal_date, source_url, importance_score and
    signal_type. Ordered by importance then recency, capped per dimension.
    """
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT signal_type, signal_date, headline, source_url,
                       importance_score
                FROM ai_company_signals
                WHERE company_name = %s
                ORDER BY importance_score DESC NULLS LAST, signal_date DESC
                """,
                (company_name,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    evidence: dict[str, list] = {dim: [] for dim in _EVIDENCE_DIMENSIONS}
    for sig_type, sig_date, headline, source_url, importance in rows:
        dim = _DISPLAY_DIMENSION.get(sig_type)
        if dim is None or len(evidence[dim]) >= per_dimension:
            continue
        evidence[dim].append(
            {
                "signal_type": sig_type,
                "date": str(sig_date) if sig_date else None,
                "headline": headline,
                "source_url": source_url,
                "importance": importance,
            }
        )
    return evidence


if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else "Anthropic"
    result = compute_financial_health(company)
    print(f"Financial Health for {company}: {result['score']} (confidence: {result['confidence']})")
    for line in result["reasoning"]:
        print(f"  {line}")

    conf = compute_confidence(company)
    print(f"\nPer-dimension confidence for {company} (overall: {conf['overall']}, "
          f"needs_review: {conf['needs_review']}):")
    for dim, info in conf["dimensions"].items():
        detail = ("data source unavailable" if info["level"] == "unknown"
                  else f"{info['evidence_count']} signals")
        print(f"  {dim}: {info['level']} ({detail})")
