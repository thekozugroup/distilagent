"""DistilAgent HTTP API — FastAPI app with SSE event stream."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from . import db as dbmod
from .runner import RunManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
_LOG = logging.getLogger("distilagent.api")

DATA_DIR = Path(os.environ.get("DISTILAGENT_DATA", "/data"))
DB_PATH = DATA_DIR / "meta.db"
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
WEB_DIST = REPO_ROOT / "web" / "dist"

# Try to import the module-level pool defaults so the UI can show them.
import sys
sys.path.insert(0, str(SCRIPTS_DIR))
try:
    from pools import GENERALIST_POOL, JUDGE_POOL, CODE_SPECIALIST, REASONING_SPECIALIST
except Exception:  # noqa: BLE001
    GENERALIST_POOL = JUDGE_POOL = []
    CODE_SPECIALIST = REASONING_SPECIALIST = ""


app = FastAPI(title="DistilAgent API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_runner: Optional[RunManager] = None


def get_runner() -> RunManager:
    global _runner
    if _runner is None:
        _runner = RunManager(DATA_DIR, DB_PATH, SCRIPTS_DIR)
    return _runner


class RunConfig(BaseModel):
    provider: str = "openrouter"
    base_url: Optional[str] = None
    model: Optional[str] = "minimax/minimax-m2.5:free"
    use_pool: bool = True
    no_routing: bool = True
    teacher_pool: Optional[List[str]] = None
    judge_pool: Optional[List[str]] = None
    n_hf: int = 50
    hf_seed: int = 2026
    no_agentic: bool = False
    num_judges: int = 3
    max_iterations: int = 2
    convergence_k: int = 2
    max_concurrency: int = 3
    rpm: int = 8
    rpd: int = 200
    temperature: float = 0.5
    top_p: float = 0.95
    repetition_penalty: float = 1.2
    max_tokens: int = 4096
    enable_thinking: bool = True
    llm_call_timeout: int = 300
    rng_seed: int = 42
    openrouter_api_key: Optional[str] = None  # overrides env if set


# -----------------------------------------------------------------------------
# health & pools
# -----------------------------------------------------------------------------

@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "data_dir": str(DATA_DIR)}


@app.get("/api/pools")
def pools() -> Dict[str, Any]:
    return {
        "generalist_pool": GENERALIST_POOL,
        "judge_pool": JUDGE_POOL,
        "code_specialist": CODE_SPECIALIST,
        "reasoning_specialist": REASONING_SPECIALIST,
        "defaults": RunConfig().model_dump(),
    }


# -----------------------------------------------------------------------------
# runs
# -----------------------------------------------------------------------------

@app.get("/api/runs")
def list_runs() -> List[Dict[str, Any]]:
    return dbmod.list_runs(get_runner().conn)


@app.post("/api/runs", status_code=201)
async def create_run(cfg: RunConfig) -> Dict[str, Any]:
    run_id = await get_runner().start(cfg.model_dump())
    return {"run_id": run_id}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    run = dbmod.get_run(get_runner().conn, run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Dict[str, Any]:
    ok = await get_runner().cancel(run_id)
    if not ok:
        raise HTTPException(404, "run not active")
    return {"cancelled": True}


@app.get("/api/runs/{run_id}/download")
def download_run(run_id: str) -> FileResponse:
    run = dbmod.get_run(get_runner().conn, run_id)
    if not run or not run.get("output_path"):
        raise HTTPException(404, "no output")
    p = Path(run["output_path"])
    if not p.exists():
        raise HTTPException(404, "output file missing")
    return FileResponse(p, filename=f"distilagent_{run_id}.jsonl", media_type="application/x-ndjson")


@app.get("/api/runs/{run_id}/samples")
def run_samples(run_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    # Return a trimmed payload (no full reasoning_trace) for list views.
    rows = get_runner().samples_for_run(run_id, limit=limit)
    out = []
    for r in rows:
        out.append({
            "id": r.get("id"),
            "category": r.get("category"),
            "instruction": (r.get("instruction") or "")[:400],
            "autoreason_iterations": r.get("autoreason_iterations"),
            "autoreason_converged": r.get("autoreason_converged"),
            "total_calls": r.get("total_calls"),
            "elapsed_seconds": r.get("elapsed_seconds"),
            "model_name": r.get("model_name"),
            "generation_chars": len(r.get("generation") or ""),
            "reasoning_total_chars": r.get("reasoning_total_chars"),
        })
    return out


@app.get("/api/runs/{run_id}/samples/{sample_id}")
def sample_detail(run_id: str, sample_id: str) -> Dict[str, Any]:
    s = get_runner().sample_by_id(run_id, sample_id)
    if not s:
        raise HTTPException(404, "sample not found")
    return s


# -----------------------------------------------------------------------------
# SSE event stream
# -----------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/sse")
async def sse(run_id: str, last_event_id: int = 0):
    runner = get_runner()
    run = dbmod.get_run(runner.conn, run_id)
    if not run:
        raise HTTPException(404, "run not found")

    async def generator():
        nonlocal last_event_id
        # Replay anything that landed before the client connected.
        backlog = dbmod.events_since(runner.conn, run_id, last_event_id)
        for e in backlog:
            last_event_id = e["id"]
            yield {"id": str(e["id"]), "event": e["kind"], "data": json.dumps(e["payload"])}
        # Poll for new events.
        while True:
            await asyncio.sleep(1.0)
            new = dbmod.events_since(runner.conn, run_id, last_event_id)
            for e in new:
                last_event_id = e["id"]
                yield {"id": str(e["id"]), "event": e["kind"], "data": json.dumps(e["payload"])}
            run = dbmod.get_run(runner.conn, run_id) or {}
            if run.get("status") in ("done", "failed", "cancelled"):
                # One final terminal event.
                yield {"event": "terminal", "data": json.dumps({"status": run.get("status")})}
                break

    return EventSourceResponse(generator())


# -----------------------------------------------------------------------------
# static frontend (mount LAST so /api/* takes precedence)
# -----------------------------------------------------------------------------

if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")


@app.get("/")
def root_stub() -> Dict[str, Any]:
    # Only reached if the static mount is absent (dev mode).
    return {
        "service": "distilagent",
        "version": "0.1.0",
        "note": "web bundle not present — build `web/` or hit /api/* endpoints directly",
    }
