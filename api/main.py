"""FastAPI application wrapping the LangGraph intelligence brief generation."""

import sys
import os
import json
from pathlib import Path
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import psycopg2.extras
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.supervisor import generate_brief
from agents.scoring import get_evidence
from collector.db import get_connection

# Load company names from seed file
SEED_FILE = Path(__file__).parent.parent / "collector" / "seed_companies.json"
with open(SEED_FILE) as f:
    SEED_COMPANIES = json.load(f)
    COMPANY_NAMES = {c["name"] for c in SEED_COMPANIES}

# FastAPI app
app = FastAPI(
    title="AI Vendor Intelligence Platform",
    description="Generates one-page intelligence briefs on AI vendors using multi-agent LangGraph orchestration",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------


class BriefRequest(BaseModel):
    """Request to generate a brief for a company."""

    company: str


class BriefResponse(BaseModel):
    """Response containing generated brief."""

    company: str
    brief: str
    status: str
    review_status: str = "auto_ok"
    needs_review: bool = False
    confidence: dict | None = None
    evidence: dict | None = None


class ReviewItem(BaseModel):
    """A brief awaiting human review."""

    id: int
    company: str
    brief: str
    review_status: str
    confidence: dict | None = None
    cache_date: str


class ReviewQueueResponse(BaseModel):
    """List of briefs flagged for human review."""

    items: list[ReviewItem]
    count: int


class ReviewActionRequest(BaseModel):
    """A reviewer's decision on a flagged brief."""

    notes: str | None = None
    brief: str | None = None  # optional corrected brief text


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agents: int
    model: str


class CompaniesResponse(BaseModel):
    """List of available companies."""

    companies: list[str]
    count: int


# ---------------------------------------------------------------------------
# Brief Caching Helpers
# ---------------------------------------------------------------------------


def get_cached_brief(company: str) -> dict | None:
    """Retrieve today's cached brief as a dict, or None if not in cache.

    Returns keys: brief, review_status, confidence.
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT brief, review_status, confidence FROM brief_cache
                    WHERE company_name = %s AND cache_date = %s
                    LIMIT 1
                    """,
                    (company, date.today()),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "brief": row[0],
                    "review_status": row[1] or "auto_ok",
                    "confidence": row[2],
                }
    except Exception as e:
        print(f"WARNING: Cache lookup failed for {company}: {e}")
        return None


def save_brief_cache(
    company: str,
    brief: str,
    review_status: str = "auto_ok",
    confidence: dict | None = None,
) -> None:
    """Store brief in cache for today, including review status and confidence."""
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO brief_cache
                        (company_name, brief, cache_date, review_status, confidence)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (company_name, cache_date) DO UPDATE SET
                        brief = EXCLUDED.brief,
                        review_status = EXCLUDED.review_status,
                        confidence = EXCLUDED.confidence
                    """,
                    (
                        company,
                        brief,
                        date.today(),
                        review_status,
                        psycopg2.extras.Json(confidence) if confidence else None,
                    ),
                )
    except Exception as e:
        print(f"WARNING: Cache save failed for {company}: {e}")


# ---------------------------------------------------------------------------
# Root & Health Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "message": "AI Vendor Intelligence Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "openapi": "/openapi.json",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agents": 5,
        "model": "amazon.nova-micro-v1:0",
    }


# ---------------------------------------------------------------------------
# Companies Endpoint
# ---------------------------------------------------------------------------


@app.get("/companies", response_model=CompaniesResponse)
async def get_companies():
    """Get list of available companies for brief generation."""
    return {
        "companies": sorted(list(COMPANY_NAMES)),
        "count": len(COMPANY_NAMES),
    }


# ---------------------------------------------------------------------------
# Brief Generation Endpoint
# ---------------------------------------------------------------------------


@app.post("/brief", response_model=BriefResponse)
async def generate_brief_endpoint(request: BriefRequest):
    """Generate an intelligence brief for a company.

    Takes ~20-30 seconds on first request; cached responses are near-instant.
    """
    company = request.company.strip()

    # Validate input
    if not company:
        raise HTTPException(status_code=400, detail="Company name required")

    print(f"[API] Generating brief for: {company}")

    # Evidence is fetched fresh from the current signals on every request (one
    # cheap query) so briefs always show up-to-date, verifiable sources.
    def _evidence():
        try:
            return get_evidence(company)
        except Exception as e:
            print(f"WARNING: Evidence lookup failed for {company}: {e}")
            return None

    # Check cache first
    cached = get_cached_brief(company)
    if cached:
        print(f"[API] Cache hit for: {company}")
        review_status = cached["review_status"]
        return {
            "company": company,
            "brief": cached["brief"],
            "status": "cached",
            "review_status": review_status,
            "needs_review": review_status == "needs_review",
            "confidence": cached["confidence"],
            "evidence": _evidence(),
        }

    try:
        # Generate the brief using LangGraph (returns brief + confidence)
        result = generate_brief(company)
        brief = result["brief"]
        review_status = result["review_status"]

        # Save to cache, including confidence + review status. Serve-with-banner
        # mode: flagged briefs are still cached and returned, just annotated.
        save_brief_cache(company, brief, review_status, result["confidence"])

        # Check if company was in seed (informational only)
        if company not in COMPANY_NAMES:
            print(
                f"[API] WARNING: {company} not in seed companies, but brief generated with available data"
            )

        return {
            "company": company,
            "brief": brief,
            "status": "success",
            "review_status": review_status,
            "needs_review": result["needs_review"],
            "confidence": result["confidence"],
            "evidence": _evidence(),
        }

    except Exception as exc:
        print(f"[API] ERROR generating brief for {company}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate brief: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Human Review Endpoints (solo reviewer)
# ---------------------------------------------------------------------------


@app.get("/review-queue", response_model=ReviewQueueResponse)
async def review_queue():
    """List briefs flagged for human review (review_status = 'needs_review')."""
    try:
        conn = get_connection()
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, company_name, brief, review_status,
                           confidence, cache_date
                    FROM brief_cache
                    WHERE review_status = 'needs_review'
                    ORDER BY cache_date DESC, company_name ASC
                    """
                )
                rows = cur.fetchall()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Review queue lookup failed: {exc}")

    items = [
        {
            "id": r["id"],
            "company": r["company_name"],
            "brief": r["brief"],
            "review_status": r["review_status"],
            "confidence": r["confidence"],
            "cache_date": str(r["cache_date"]),
        }
        for r in rows
    ]
    return {"items": items, "count": len(items)}


@app.post("/review/{brief_id}", response_model=BriefResponse)
async def review_brief(brief_id: int, action: ReviewActionRequest):
    """Approve (and optionally correct) a flagged brief.

    Marks it 'reviewed', records the reviewer's notes and the review time, and
    optionally overwrites the brief text with a human-corrected version.
    """
    try:
        conn = get_connection()
        with conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT company_name, brief FROM brief_cache WHERE id = %s",
                    (brief_id,),
                )
                existing = cur.fetchone()
                if not existing:
                    raise HTTPException(status_code=404, detail="Brief not found")

                new_brief = action.brief if action.brief is not None else existing["brief"]
                cur.execute(
                    """
                    UPDATE brief_cache
                    SET review_status = 'reviewed',
                        brief = %s,
                        review_notes = %s,
                        reviewed_at = NOW()
                    WHERE id = %s
                    """,
                    (new_brief, action.notes, brief_id),
                )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Review update failed: {exc}")

    return {
        "company": existing["company_name"],
        "brief": new_brief,
        "status": "reviewed",
        "review_status": "reviewed",
        "needs_review": False,
        "confidence": None,
    }


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors."""
    return HTTPException(status_code=400, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
