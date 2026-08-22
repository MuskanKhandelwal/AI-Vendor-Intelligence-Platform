"""Audit stored signals for wrong-entity contamination and purge the bad rows.

Collectors gained entity verification (see entity_filter.py) only after signals
had already been collected by name search alone, so the database still holds
rows about namesake organisations — a $28M defence contract awarded to Cohere
Technologies stored under the AI vendor Cohere, executive changes at Cohere
Health, and quantum-physics papers matched on the stemmed word "coherence".
Those rows are cited as sourced evidence in briefs, so they need removing.

Only name-searched signals are audited. Rows from the GitHub collector are keyed
to a `github_org` and rows from the EDGAR collector to a CIK, so neither can be
a name collision; they are identified by their raw_data keys and skipped.

Usage:
    python collector/audit_entities.py                  # dry run, report only
    python collector/audit_entities.py --company Cohere # limit to one company
    python collector/audit_entities.py --apply          # delete the bad rows
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

BACKUP_DIR = Path(__file__).parent.parent / "backups"

sys.path.insert(0, os.path.dirname(__file__))
import db
from entity_filter import check_entity, load_company_index

# raw_data keys that mark a signal as having come from a name search, and thus
# vulnerable to namesake collisions.
NEWS_KEYS = ("feed_title", "original_title")
ARXIV_KEYS = ("arxiv_id",)

# Mirrors arxiv_collector; applied to rows whose raw_data records categories.
ALLOWED_CATEGORY_PREFIXES = ("cs.",)
ALLOWED_CATEGORIES = frozenset({"stat.ML", "eess.AS", "eess.IV"})


def _source_of(raw_data: dict | None) -> str | None:
    """Return 'news', 'arxiv', or None for signals not derived from a name."""
    if not raw_data:
        return None
    if any(k in raw_data for k in ARXIV_KEYS):
        return "arxiv"
    if any(k in raw_data for k in NEWS_KEYS):
        return "news"
    return None


def _category_verdict(raw_data: dict) -> str | None:
    """Reject reason if a paper's recorded arXiv category is off-topic.

    Returns None when the category is fine or when the row predates category
    recording, in which case the name check is the only available test.
    """
    primary = raw_data.get("primary_category")
    candidates = [primary] if primary else raw_data.get("categories") or []
    if not candidates:
        return None
    if any(
        c in ALLOWED_CATEGORIES or c.startswith(ALLOWED_CATEGORY_PREFIXES)
        for c in candidates
        if c
    ):
        return None
    return f"off-topic arXiv category ({primary or ', '.join(candidates)})"


def find_contaminated(company_filter: str | None = None) -> list[dict]:
    """Return every stored signal that fails its company's entity check."""
    index = load_company_index()

    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, company_name, signal_type, signal_date, headline,
                       summary, raw_data
                FROM ai_company_signals
                ORDER BY company_name, signal_date DESC
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    bad = []
    for sig_id, company_name, sig_type, sig_date, headline, summary, raw in rows:
        if company_filter and company_name != company_filter:
            continue

        company = index.get(company_name)
        if company is None:
            continue  # not in the seed list; nothing to check against

        source = _source_of(raw)
        if source is None:
            continue  # GitHub/EDGAR row, keyed to an authoritative identifier

        reason = None
        if source == "arxiv":
            reason = _category_verdict(raw or {})
        if reason is None:
            verdict = check_entity(company, headline, summary)
            if verdict.ok:
                continue
            reason = verdict.reason

        bad.append({
            "id": sig_id,
            "company": company_name,
            "signal_type": sig_type,
            "date": sig_date,
            "headline": headline,
            "reason": reason,
        })

    return bad


def backup(signal_ids: list[int]) -> Path:
    """Dump full rows for the given signals to a timestamped JSON file.

    The purge is irreversible, so every flagged row is written out first. If a
    rejection later proves wrong, the row can be re-inserted from this file.
    """
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, company_name, ticker, signal_type, signal_date,
                       headline, summary, source_url, importance_score,
                       raw_data, langfuse_trace_id, created_at
                FROM ai_company_signals
                WHERE id = ANY(%s)
                """,
                (signal_ids,),
            )
            columns = [d[0] for d in cur.description]
            rows = [
                {
                    col: (val.isoformat() if hasattr(val, "isoformat") else val)
                    for col, val in zip(columns, row)
                }
                for row in cur.fetchall()
            ]
    finally:
        conn.close()

    BACKUP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = BACKUP_DIR / f"wrong_entity_signals_{stamp}.json"
    path.write_text(json.dumps(rows, indent=2))
    return path


def purge(signal_ids: list[int], companies: list[str]) -> tuple[int, int]:
    """Delete the given signals and invalidate affected cached briefs.

    Cached briefs quote the deleted signals, so they are dropped too; the next
    request regenerates them from the cleaned data. Returns (signals_deleted,
    briefs_invalidated).
    """
    if not signal_ids:
        return 0, 0

    conn = db.get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM ai_company_signals WHERE id = ANY(%s)",
                    (signal_ids,),
                )
                deleted = cur.rowcount
                cur.execute(
                    "DELETE FROM brief_cache WHERE company_name = ANY(%s)",
                    (companies,),
                )
                invalidated = cur.rowcount
    finally:
        conn.close()

    return deleted, invalidated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company", help="audit a single company by name")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="delete the contaminated rows (default is a dry run)",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="skip the JSON backup written before deleting",
    )
    args = parser.parse_args()

    bad = find_contaminated(args.company)

    if not bad:
        print("No wrong-entity signals found.")
        return

    by_company: dict[str, list[dict]] = defaultdict(list)
    for row in bad:
        by_company[row["company"]].append(row)

    print(f"Found {len(bad)} wrong-entity signal(s) across "
          f"{len(by_company)} company/companies:\n")
    for company, rows in sorted(by_company.items()):
        print(f"{company} ({len(rows)}):")
        for r in rows:
            print(f"  [{r['signal_type']} {r['date']}] {(r['headline'] or '')[:80]}")
            print(f"      reason: {r['reason']}")
        print()

    if not args.apply:
        print("Dry run — nothing deleted. Re-run with --apply to remove these "
              "rows and invalidate the affected cached briefs.")
        return

    if not args.no_backup:
        path = backup([r["id"] for r in bad])
        print(f"Backed up {len(bad)} row(s) to {path}\n")

    deleted, invalidated = purge(
        [r["id"] for r in bad], sorted(by_company.keys())
    )
    print(f"Deleted {deleted} signal(s) and invalidated {invalidated} cached "
          f"brief(s). Affected companies will regenerate on next request.")


if __name__ == "__main__":
    main()
