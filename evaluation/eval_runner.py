"""Deterministic evaluation harness for generated vendor briefs.

Checks a brief against the evidence that was actually collected for that
company, rather than asking another LLM whether it looks right. This mirrors
the rest of the project: where a rule can be written down, we write the rule
instead of trusting a model to judge.

Two severities:
  FAIL  a contract the brief must satisfy (structure, score fidelity, honest
        UNKNOWNs, no banned filler). A FAIL is a real defect.
  WARN  a heuristic grounding check (dates, dollar amounts). These can flag
        legitimate aggregates the model derived from tool output rather than
        copied from a headline, so they need a human glance, not a red light.

Usage:
    python evaluation/eval_runner.py                     # default company set
    python evaluation/eval_runner.py Anthropic OpenAI    # specific companies
    python evaluation/eval_runner.py --emit-labels       # append claims to
                                                         # labeled_briefs.csv
"""

import csv
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.scoring import compute_confidence, compute_financial_health, get_evidence
from agents.supervisor import generate_brief

LABELS_FILE = Path(__file__).parent / "labeled_briefs.csv"
DEFAULT_COMPANIES = ["Anthropic", "OpenAI", "Scale AI"]

# Section header -> the confidence dimension that backs it.
SECTIONS = {
    "FINANCIAL HEALTH": "financial",
    "TECHNOLOGY MOMENTUM": "technology",
    "NEWS & SENTIMENT": "news",
    "PERSONNEL STABILITY": "personnel",
    "COMPETITIVE POSITION": "competitive",
    "EXECUTIVE SUMMARY": None,  # synthesises the others, has no own dimension
}

# Generic filler the synthesis prompt explicitly forbids. If a phrase would be
# true of any AI vendor, it carries no procurement signal.
BANNED_PHRASES = [
    "prominent player",
    "key competitor",
    "well positioned",
    "well-positioned",
    "promising outlook",
    "strong market presence",
    "compelling option",
    "committed to responsible ai",
    "leading provider",
    "at the forefront",
    "cutting-edge",
]

_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_AMOUNT_RE = re.compile(r"\$\s?(\d+(?:\.\d+)?)\s*(million|billion|[MB])\b", re.I)
_SCORE_RE = re.compile(r"(\d{1,3})\s*/\s*100")


# ---------------------------------------------------------------------------
# Brief parsing
# ---------------------------------------------------------------------------


def split_sections(brief: str) -> dict[str, str]:
    """Split a brief into {SECTION NAME: body text}.

    Tolerates the markdown bold the synthesiser tends to emit (**HEADER:**).
    """
    positions = []
    for name in SECTIONS:
        match = re.search(rf"\*{{0,2}}{re.escape(name)}\*{{0,2}}\s*:?", brief)
        if match:
            positions.append((match.start(), match.end(), name))

    positions.sort()
    sections = {}
    for i, (_, body_start, name) in enumerate(positions):
        body_end = positions[i + 1][0] if i + 1 < len(positions) else len(brief)
        sections[name] = brief[body_start:body_end].strip()
    return sections


def _normalize_amount(value: str, unit: str) -> float:
    """Return a dollar amount in millions, so $1.2B and $1200M compare equal."""
    amount = float(value)
    return amount * 1000 if unit.lower().startswith("b") else amount


def _evidence_corpus(company_name: str) -> tuple[set[str], set[float], str]:
    """Return (dates, amounts in millions, all headline text) from real signals."""
    evidence = get_evidence(company_name, per_dimension=500)

    dates: set[str] = set()
    amounts: set[float] = set()
    text_parts: list[str] = []

    for items in evidence.values():
        for item in items:
            if item["date"]:
                dates.add(item["date"])
            headline = item["headline"] or ""
            text_parts.append(headline)
            for value, unit in _AMOUNT_RE.findall(headline):
                amounts.add(_normalize_amount(value, unit))

    return dates, amounts, " ".join(text_parts).lower()


# ---------------------------------------------------------------------------
# Checks — each returns a list of failure strings (empty means it passed)
# ---------------------------------------------------------------------------


def check_structure(sections: dict) -> list[str]:
    """Every required section must be present."""
    return [f"missing section: {name}" for name in SECTIONS if name not in sections]


def check_score_fidelity(company_name: str, sections: dict) -> list[str]:
    """The financial score in the brief must equal the deterministic rubric.

    This is the regression test for the class of bug where the LLM invents its
    own number instead of reporting compute_financial_health()'s.
    """
    body = sections.get("FINANCIAL HEALTH")
    if body is None:
        return []

    expected = compute_financial_health(company_name)["score"]
    match = _SCORE_RE.search(body)

    if expected is None:
        if match:
            return [
                f"claims financial score {match.group(1)}/100 but there is no "
                "data to compute one (expected UNKNOWN)"
            ]
        return []

    if not match:
        return [f"no financial score stated, expected {expected}/100"]

    claimed = int(match.group(1))
    if claimed != expected:
        return [f"financial score {claimed}/100 does not match rubric {expected}/100"]
    return []


def check_unknown_discipline(company_name: str, sections: dict) -> list[str]:
    """Dimensions with no evidence must say UNKNOWN, not write prose anyway.

    Catches the failure mode where a data source is down and the model fills
    the gap with confident-sounding generalities.
    """
    confidence = compute_confidence(company_name)
    failures = []

    for section, dimension in SECTIONS.items():
        if dimension is None or section not in sections:
            continue

        info = confidence["dimensions"].get(dimension) or {}
        has_evidence = info.get("level") != "unknown" and info.get("evidence_count")
        if has_evidence:
            continue

        if "unknown" not in sections[section].lower():
            failures.append(
                f"{section}: no evidence collected ({dimension} is "
                f"'{info.get('level')}') but the section states findings "
                "instead of UNKNOWN"
            )
    return failures


def check_filler(sections: dict) -> list[str]:
    """No generic phrasing that would be true of any vendor."""
    failures = []
    for section, body in sections.items():
        lowered = body.lower()
        for phrase in BANNED_PHRASES:
            if phrase in lowered:
                failures.append(f"{section}: banned filler phrase '{phrase}'")
    return failures


def check_date_grounding(brief: str, dates: set[str]) -> list[str]:
    """Every explicit date cited should trace to a collected signal."""
    return [
        f"cites date {d} that appears in no collected signal"
        for d in sorted(set(_ISO_DATE_RE.findall(brief)))
        if d not in dates
    ]


def check_amount_grounding(brief: str, amounts: set[float]) -> list[str]:
    """Every dollar figure cited should trace to a collected headline."""
    failures = []
    for value, unit in _AMOUNT_RE.findall(brief):
        millions = _normalize_amount(value, unit)
        if millions not in amounts:
            failures.append(
                f"cites ${value}{unit} that appears in no collected headline"
            )
    return failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def evaluate(company_name: str, brief: str) -> dict:
    """Run every check against one brief. Returns a result dict."""
    sections = split_sections(brief)
    dates, amounts, _ = _evidence_corpus(company_name)

    return {
        "company": company_name,
        "fail": {
            "structure": check_structure(sections),
            "score_fidelity": check_score_fidelity(company_name, sections),
            "unknown_discipline": check_unknown_discipline(company_name, sections),
            "filler": check_filler(sections),
        },
        "warn": {
            "date_grounding": check_date_grounding(brief, dates),
            "amount_grounding": check_amount_grounding(brief, amounts),
        },
        "sections": sections,
    }


def _print_result(result: dict) -> tuple[int, int]:
    """Print one company's result. Returns (fail_count, warn_count)."""
    fails = sum(len(v) for v in result["fail"].values())
    warns = sum(len(v) for v in result["warn"].values())

    status = "PASS" if fails == 0 else "FAIL"
    print(f"\n{status}  {result['company']}  ({fails} failure(s), {warns} warning(s))")

    for check, issues in result["fail"].items():
        for issue in issues:
            print(f"    FAIL [{check}] {issue}")
    for check, issues in result["warn"].items():
        for issue in issues:
            print(f"    WARN [{check}] {issue}")

    return fails, warns


def emit_label_rows(result: dict) -> int:
    """Append this brief's section claims to labeled_briefs.csv for human review.

    Fills every column except `accurate`, which a human sets to y/n. This turns
    the CSV from an empty placeholder into a real labelling queue.
    """
    existing = LABELS_FILE.exists() and LABELS_FILE.stat().st_size > 0
    written = 0

    with open(LABELS_FILE, "a", newline="") as handle:
        writer = csv.writer(handle)
        if not existing:
            writer.writerow(
                ["company", "section", "finding", "accurate", "source_url", "notes"]
            )
        for section, body in result["sections"].items():
            finding = " ".join(body.split())[:300]
            if finding:
                writer.writerow([result["company"], section, finding, "", "", ""])
                written += 1
    return written


def run(companies: list[str], emit_labels: bool = False) -> int:
    """Evaluate each company. Returns the process exit code."""
    total_fails = 0
    total_warns = 0
    labelled = 0

    for company in companies:
        try:
            brief = generate_brief(company)["brief"]
        except Exception as exc:
            print(f"\nERROR  {company}: could not generate brief: {exc}")
            total_fails += 1
            continue

        result = evaluate(company, brief)
        fails, warns = _print_result(result)
        total_fails += fails
        total_warns += warns

        if emit_labels:
            labelled += emit_label_rows(result)

    print(f"\n{'=' * 70}")
    print(
        f"{len(companies)} brief(s) evaluated — "
        f"{total_fails} failure(s), {total_warns} warning(s)."
    )
    if emit_labels:
        print(f"{labelled} claim row(s) appended to {LABELS_FILE.name} for labelling.")

    return 1 if total_fails else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(
        run(
            companies=args or DEFAULT_COMPANIES,
            emit_labels="--emit-labels" in sys.argv,
        )
    )
