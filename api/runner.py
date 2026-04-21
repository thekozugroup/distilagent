"""Run lifecycle: spawn `scripts/build_dataset.py` as a subprocess and
tail its log, parsing ✓/✗/[plan]/[pool]/Traceback lines into structured
events in SQLite so the API can stream them over SSE.

Each run gets:
    /data/output/<run_id>.jsonl        — append-only row output
    /data/checkpoints/<run_id>.json    — resume checkpoint
    /data/logs/<run_id>.log            — stdout+stderr of the subprocess
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import db as dbmod

_LOG = logging.getLogger(__name__)

# Line parsers (mirror the format printed by scripts/build_dataset.py)
_RE_PLAN = re.compile(r"\[plan\]\s+total=(\d+)\s+done=(\d+)\s+remaining=(\d+)")
_RE_SAMPLE_START = re.compile(
    r"^\[(\d+)/(\d+)\]\s+(\S+)\s+\[(\w+)\](?:\s+\[(route=\S+)\s+teacher=(\S+)\])?\s+\"(.*)$"
)
_RE_SAMPLE_OK = re.compile(
    r"^\s*✓\s+iters=(\d+)\s+converged=(True|False)\s+calls=(\d+)\s+([\d.]+)s"
)
_RE_SAMPLE_FAIL = re.compile(r"^\s*✗\s+FAIL:\s+(.*)$")
_RE_POOL_WARM = re.compile(r"\[pool\]\s+warming (\d+) LLM clients")
_RE_DONE = re.compile(r"\[done\]\s+completed\s+(\d+)\s+failed\s+(\d+)")


class RunManager:
    """Process supervisor + log parser + event fanout."""

    def __init__(self, data_dir: Path, db_path: Path, scripts_dir: Path) -> None:
        self.data_dir = data_dir
        self.scripts_dir = scripts_dir
        self.conn = dbmod.get_conn(db_path)
        dbmod.init(self.conn)
        self._procs: Dict[str, asyncio.subprocess.Process] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cached_sample: Dict[str, Dict[str, Any]] = {}

        (data_dir / "output").mkdir(parents=True, exist_ok=True)
        (data_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)

    # -- public API ---------------------------------------------------------

    async def start(self, config: Dict[str, Any]) -> str:
        run_id = dbmod.new_run(self.conn, config)
        out_path = self.data_dir / "output" / f"{run_id}.jsonl"
        ckpt_path = self.data_dir / "checkpoints" / f"{run_id}.json"
        log_path = self.data_dir / "logs" / f"{run_id}.log"
        dbmod.update_run(
            self.conn, run_id,
            output_path=str(out_path),
            checkpoint_path=str(ckpt_path),
            log_path=str(log_path),
        )
        cmd = self._build_cmd(config, out_path, ckpt_path)
        env = {**os.environ}
        # Per-run overrides flow through env.
        if config.get("openrouter_api_key"):
            env["OPENROUTER_API_KEY"] = config["openrouter_api_key"]
        env.setdefault("LLM_CALL_TIMEOUT", str(config.get("llm_call_timeout", 300)))

        log_fh = open(log_path, "w", buffering=1)  # line-buffered
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=self.scripts_dir.parent,
        )
        dbmod.update_run(
            self.conn, run_id,
            status="running",
            started_at=time.time(),
            pid=proc.pid,
        )
        dbmod.append_event(self.conn, run_id, "info", {"msg": "run started", "pid": proc.pid, "cmd": cmd})
        self._procs[run_id] = proc
        self._tasks[run_id] = asyncio.create_task(self._tail(run_id, proc, log_fh))
        return run_id

    async def cancel(self, run_id: str) -> bool:
        proc = self._procs.get(run_id)
        if not proc:
            return False
        try:
            proc.send_signal(signal.SIGTERM)
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        dbmod.update_run(self.conn, run_id, status="cancelled", finished_at=time.time())
        dbmod.append_event(self.conn, run_id, "info", {"msg": "cancelled"})
        return True

    # -- subprocess log tail + parser --------------------------------------

    def _build_cmd(
        self, config: Dict[str, Any], out_path: Path, ckpt_path: Path
    ) -> List[str]:
        py = sys.executable
        script = self.scripts_dir / "build_dataset.py"
        cmd = [py, str(script),
               "--out", str(out_path),
               "--checkpoint", str(ckpt_path)]
        simple = {
            "provider": "--provider",
            "base_url": "--base-url",
            "model": "--model",
            "num_judges": "--num-judges",
            "max_iterations": "--max-iterations",
            "convergence_k": "--convergence-k",
            "max_concurrency": "--max-concurrency",
            "rpm": "--rpm",
            "rpd": "--rpd",
            "temperature": "--temperature",
            "top_p": "--top-p",
            "repetition_penalty": "--repetition-penalty",
            "max_tokens": "--max-tokens",
            "n_hf": "--n-hf",
            "hf_seed": "--hf-seed",
            "rng_seed": "--rng-seed",
        }
        for k, flag in simple.items():
            v = config.get(k)
            if v is not None:
                cmd += [flag, str(v)]
        if config.get("enable_thinking"):
            cmd.append("--enable-thinking")
        if config.get("no_agentic"):
            cmd.append("--no-agentic")
        if config.get("use_pool"):
            cmd.append("--use-pool")
        if config.get("no_routing"):
            cmd.append("--no-routing")
        if config.get("teacher_pool"):
            cmd += ["--teacher-pool", ",".join(config["teacher_pool"])]
        if config.get("judge_pool"):
            cmd += ["--judge-pool", ",".join(config["judge_pool"])]
        return cmd

    async def _tail(
        self,
        run_id: str,
        proc: asyncio.subprocess.Process,
        log_fh,
    ) -> None:
        try:
            assert proc.stdout is not None
            while True:
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                log_fh.write(line + "\n")
                log_fh.flush()
                self._parse_line(run_id, line)
            rc = await proc.wait()
            status = "done" if rc == 0 else "failed"
            dbmod.update_run(
                self.conn, run_id,
                status=status,
                finished_at=time.time(),
                error=None if rc == 0 else f"exit code {rc}",
            )
            dbmod.append_event(self.conn, run_id, "done", {"exit_code": rc, "status": status})
        except Exception as exc:  # noqa: BLE001
            _LOG.exception("tail error for run %s", run_id)
            dbmod.update_run(self.conn, run_id, status="failed", error=str(exc))
            dbmod.append_event(self.conn, run_id, "error", {"msg": str(exc)})
        finally:
            log_fh.close()
            self._procs.pop(run_id, None)
            self._tasks.pop(run_id, None)

    def _parse_line(self, run_id: str, line: str) -> None:
        stripped = line.strip()
        if m := _RE_PLAN.search(stripped):
            total, done, remaining = map(int, m.groups())
            dbmod.update_run(
                self.conn, run_id,
                total_planned=total, completed=done,
            )
            dbmod.append_event(
                self.conn, run_id, "plan",
                {"total": total, "done": done, "remaining": remaining},
            )
            return
        if m := _RE_POOL_WARM.search(stripped):
            dbmod.append_event(
                self.conn, run_id, "info",
                {"msg": "pool warming", "n_clients": int(m.group(1))},
            )
            return
        if m := _RE_SAMPLE_START.match(stripped):
            idx, total, sid, cat = m.group(1), m.group(2), m.group(3), m.group(4)
            teacher = m.group(6)
            self._cached_sample[run_id] = {
                "idx": int(idx), "total": int(total), "id": sid,
                "category": cat, "teacher": teacher,
            }
            return
        if m := _RE_SAMPLE_OK.match(stripped):
            iters, converged, calls, elapsed = m.groups()
            row = self._cached_sample.pop(run_id, {})
            payload = {
                **row,
                "status": "ok",
                "iterations": int(iters),
                "converged": converged == "True",
                "total_calls": int(calls),
                "elapsed_seconds": float(elapsed),
            }
            dbmod.append_event(self.conn, run_id, "sample", payload)
            run = dbmod.get_run(self.conn, run_id) or {}
            dbmod.update_run(
                self.conn, run_id,
                completed=(run.get("completed") or 0) + 1,
            )
            return
        if m := _RE_SAMPLE_FAIL.match(stripped):
            err = m.group(1)
            row = self._cached_sample.pop(run_id, {})
            dbmod.append_event(
                self.conn, run_id, "sample",
                {**row, "status": "fail", "error": err},
            )
            run = dbmod.get_run(self.conn, run_id) or {}
            dbmod.update_run(
                self.conn, run_id,
                failed=(run.get("failed") or 0) + 1,
            )
            return
        if m := _RE_DONE.search(stripped):
            completed, failed = int(m.group(1)), int(m.group(2))
            dbmod.update_run(
                self.conn, run_id,
                completed=completed, failed=failed,
            )
            return
        if stripped.startswith("Traceback"):
            dbmod.append_event(
                self.conn, run_id, "error",
                {"msg": "traceback started", "line": stripped[:240]},
            )

    # -- helpers for API ---------------------------------------------------

    def samples_for_run(self, run_id: str, limit: int = 500) -> List[Dict[str, Any]]:
        run = dbmod.get_run(self.conn, run_id)
        if not run or not run.get("output_path"):
            return []
        p = Path(run["output_path"])
        if not p.exists():
            return []
        out = []
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:  # noqa: BLE001
                    continue
                if len(out) >= limit:
                    break
        return out

    def sample_by_id(self, run_id: str, sample_id: str) -> Optional[Dict[str, Any]]:
        for r in self.samples_for_run(run_id, limit=10**6):
            if r.get("id") == sample_id:
                return r
        return None
