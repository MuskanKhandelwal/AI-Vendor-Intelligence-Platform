"""FastAPI application wrapping the LangGraph intelligence brief generation."""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.supervisor import generate_brief

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

    Takes ~20-30 seconds as it orchestrates 5 specialist agents.
    """
    company = request.company.strip()

    # Validate input
    if not company:
        raise HTTPException(status_code=400, detail="Company name required")

    print(f"[API] Generating brief for: {company}")

    try:
        # Generate the brief using LangGraph
        brief = generate_brief(company)

        # Check if company was in seed (informational only)
        if company not in COMPANY_NAMES:
            print(
                f"[API] WARNING: {company} not in seed companies, but brief generated with available data"
            )

        return {
            "company": company,
            "brief": brief,
            "status": "success",
        }

    except Exception as exc:
        print(f"[API] ERROR generating brief for {company}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate brief: {str(exc)}",
        )


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
