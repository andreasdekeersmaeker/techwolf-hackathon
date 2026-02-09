"""FastAPI server serving the role recommendations website at /jobrecommendations."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware

from config import STATIC_DIR, TEMPLATE_DIR
from models.schemas import PipelineOutput

log = logging.getLogger(__name__)

app = FastAPI(title="Layer 2 — Role Recommendations")

# Allow CORS from Layer 1 (localhost:8000) and any localhost origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class IframeHeadersMiddleware(BaseHTTPMiddleware):
    """Remove X-Frame-Options and set permissive CSP for iframe embedding."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        # Allow this site to be embedded in iframes from any origin
        response.headers["X-Frame-Options"] = "ALLOWALL"
        response.headers["Content-Security-Policy"] = "frame-ancestors *"
        return response


app.add_middleware(IframeHeadersMiddleware)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Global state — set by main.py after pipeline completes
_output: PipelineOutput | None = None


def set_pipeline_output(output: PipelineOutput) -> None:
    global _output
    _output = output


def _ctx(request: Request, **extra) -> dict:
    """Build base template context."""
    return {"request": request, "output": _output, **extra}


_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
body {
    margin: 0; padding: 0;
    background: #0a0a0a; color: #F9F9F9;
    font-family: 'Inter', sans-serif;
    display: flex; align-items: center; justify-content: center;
    min-height: 100vh;
}
.waiting {
    text-align: center; padding: 3rem;
    border: 1px solid rgba(169,88,255,0.15);
    border-radius: 16px; background: #111111;
    max-width: 480px;
}
.waiting h2 { color: #A958FF; margin-bottom: 0.75rem; }
.waiting p { color: #7B7B7B; font-size: 0.95rem; line-height: 1.6; }
.dot { animation: pulse 1.5s infinite; display: inline-block; }
.dot:nth-child(2) { animation-delay: 0.3s; }
.dot:nth-child(3) { animation-delay: 0.6s; }
@keyframes pulse { 0%,80%,100% { opacity: 0.3; } 40% { opacity: 1; } }
</style>
</head>
<body>
<div class="waiting">
    <h2>Role Recommendations</h2>
    <p>The pipeline has not run yet. Run the analysis to generate role recommendations for the designed system.</p>
    <p style="margin-top:1rem;color:#4a4a4a;font-size:0.8rem;">
        Waiting<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>
    </p>
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/jobrecommendations", response_class=HTMLResponse)
@app.get("/jobrecommendations/", response_class=HTMLResponse)
async def dashboard(request: Request):
    if _output is None:
        return HTMLResponse(_FALLBACK_HTML, status_code=200)
    return templates.TemplateResponse("dashboard.html", _ctx(request))


@app.get("/jobrecommendations/intermediate/system", response_class=HTMLResponse)
async def intermediate_system(request: Request):
    if _output is None:
        return HTMLResponse(_FALLBACK_HTML, status_code=200)
    return templates.TemplateResponse("intermediate/system.html", _ctx(request))


@app.get("/jobrecommendations/intermediate/needs", response_class=HTMLResponse)
async def intermediate_needs(request: Request):
    if _output is None:
        return HTMLResponse(_FALLBACK_HTML, status_code=200)
    return templates.TemplateResponse("intermediate/needs.html", _ctx(request))


@app.get("/jobrecommendations/intermediate/retrieval", response_class=HTMLResponse)
async def intermediate_retrieval(request: Request):
    if _output is None:
        return HTMLResponse(_FALLBACK_HTML, status_code=200)
    return templates.TemplateResponse("intermediate/retrieval.html", _ctx(request))


@app.get("/jobrecommendations/intermediate/scoring", response_class=HTMLResponse)
async def intermediate_scoring(request: Request):
    if _output is None:
        return HTMLResponse(_FALLBACK_HTML, status_code=200)
    return templates.TemplateResponse("intermediate/scoring.html", _ctx(request))


@app.get("/jobrecommendations/intermediate/clustering", response_class=HTMLResponse)
async def intermediate_clustering(request: Request):
    if _output is None:
        return HTMLResponse(_FALLBACK_HTML, status_code=200)
    return templates.TemplateResponse("intermediate/clustering.html", _ctx(request))


# JSON API for programmatic access
@app.get("/jobrecommendations/api/roster")
async def api_roster():
    if _output is None:
        return {"error": "Pipeline has not run yet"}
    return _output.roster.model_dump()


@app.get("/jobrecommendations/api/intermediate")
async def api_intermediate():
    if _output is None:
        return {"error": "Pipeline has not run yet"}
    return _output.intermediate.model_dump()
