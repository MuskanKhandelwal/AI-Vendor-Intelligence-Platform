"""Entry point for the FastAPI server."""

import sys
import os

# Ensure project root is in path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
