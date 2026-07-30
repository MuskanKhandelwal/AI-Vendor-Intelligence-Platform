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
                SELECT signal_type, signal_date, importance_score, raw_data
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
        _, signal_date, _, raw = funding_rows[0]
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
        recent_negative = [
            r for r in negative_rows if (_days_since(r[1]) or 9999) <= 90
        ]
        if recent_negative:
            score -= 30
            reasoning.append(
                f"-30: {len(recent_negative)} negative signal(s) "
                "(layoffs/bankruptcy) in last 90 days"
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


if __name__ == "__main__":
    company = sys.argv[1] if len(sys.argv) > 1 else "Anthropic"
    result = compute_financial_health(company)
    print(f"Financial Health for {company}: {result['score']} (confidence: {result['confidence']})")
    for line in result["reasoning"]:
        print(f"  {line}")
