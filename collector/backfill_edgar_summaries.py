"""One-off backfill: re-summarize existing EDGAR signals with real filing text.

The summary fix in edgar_collector.py only affects newly-collected filings —
db.insert_signal dedupes on (company, type, headline, date), and headlines were
left unchanged, so re-running the collector skips rows that already exist.
This script rewrites those existing rows in place.

Nothing is deleted. Only summary, raw_data, and langfuse_trace_id are updated;
the previous summary is preserved in raw_data["summary_before_backfill"].

Dry-run by default. Pass --apply to write.
"""

import sys
from collections import defaultdict

import edgar

import db
from edgar_collector import (
    NO_TEXT,
    _eightk_items,
    _generate_summary,
    _groq_client,
    _has_item_502,
    _item_502_text,
    _set_edgar_identity,
)

# Only EDGAR-sourced rows. The LIKE clause is what protects executive_change
# rows that came from news_collector (no SEC filing to extract text from).
_SELECT_SQL = """
SELECT id, company_name, ticker, signal_type, signal_date, headline,
       summary, raw_data
FROM   ai_company_signals
WHERE  signal_type IN ('annual_filing', 'executive_change')
  AND  source_url LIKE '%%sec.gov%%'
ORDER  BY signal_type, signal_date DESC
"""


def _fetch_rows() -> list[dict]:
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(_SELECT_SQL)
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _eightk_index(ticker: str) -> dict[str, list]:
    """Map filing_date -> 8-K filings for a ticker, for date matching."""
    index = defaultdict(list)
    for filing in edgar.Company(ticker).get_filings(form="8-K").latest(40):
        index[str(filing.filing_date)].append(filing)
    return index


def _context_for(row: dict, eightk_cache: dict) -> str:
    """Get the filing excerpt to ground this row's summary in."""
    raw = row["raw_data"] or {}

    if row["signal_type"] == "annual_filing":
        # The 10-K description was already extracted at collection time, and
        # that extraction logic is unchanged — re-fetching would be identical.
        return raw.get("description") or NO_TEXT

    ticker = row["ticker"]
    if not ticker:
        return NO_TEXT

    if ticker not in eightk_cache:
        eightk_cache[ticker] = _eightk_index(ticker)

    candidates = eightk_cache[ticker].get(str(row["signal_date"]), [])
    filing = next(
        (f for f in candidates if _has_item_502(_eightk_items(f))), None
    )
    return _item_502_text(filing) if filing else NO_TEXT


def _new_raw_data(row: dict, context: str) -> dict:
    raw = dict(row["raw_data"] or {})

    if row["signal_type"] == "annual_filing":
        raw["description"] = context
    else:
        raw["item_502_text"] = context

    raw["extraction_ok"] = context != NO_TEXT
    # setdefault: on a re-run, keep the true original, don't overwrite it
    # with an already-backfilled summary.
    raw.setdefault("summary_before_backfill", row["summary"])
    return raw


def run(apply: bool) -> None:
    _set_edgar_identity()
    groq = _groq_client()

    rows = _fetch_rows()
    print(f"{len(rows)} EDGAR-sourced row(s) to backfill.")
    print("DRY RUN — nothing will be written.\n" if not apply else "APPLYING changes.\n")

    eightk_cache: dict = {}
    updated = 0
    errors = 0

    for row in rows:
        label = f"[{row['id']}] {row['company_name']} {row['signal_type']} {row['signal_date']}"
        try:
            context = _context_for(row, eightk_cache)
            summary, trace_id = _generate_summary(
                groq,
                row["headline"],
                row["company_name"],
                row["signal_type"],
                40 if row["signal_type"] == "annual_filing" else 75,
                context=context,
            )

            print(f"{label}  extraction_ok={context != NO_TEXT}")
            print(f"  BEFORE: {row['summary']}")
            print(f"  AFTER : {summary}\n")

            if apply:
                db.update_signal_summary(
                    row["id"],
                    summary,
                    _new_raw_data(row, context),
                    langfuse_trace_id=trace_id,
                )
            updated += 1

        except Exception as exc:
            print(f"{label}  ERROR: {exc}\n")
            errors += 1

    verb = "updated" if apply else "would update"
    print(f"Done. {updated} row(s) {verb}, {errors} error(s).")
    if not apply:
        print("Re-run with --apply to write these changes.")


if __name__ == "__main__":
    run(apply="--apply" in sys.argv)
