# Copyright (c) 2026 FF EAI Robotics Ecosystem Inc. All rights reserved. Review use only; see LICENSE-FF.txt.
"""RoboColiseum Season II policy server (GenieSim v3.6): runs the task program of each episode.

    PYTHONPATH=src python -m cap_harness.g2s2.serve --port 9101 --programs examples/g2s2/programs --out runs/w9101
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any

from cap_harness.g2s2 import PACKAGE_VERSION
from cap_harness.g2s2.api import S2Api
from cap_harness.g2s2.bridge import S2Bridge
from cap_harness.g2s2.episode import Episode
from cap_harness.g2s2.motion import Limits
from cap_harness.g2s2.perception import Sam3Client
from cap_harness.g2s2.program import run_program
from cap_harness.g2s2.tasks import resolve_task

logger = logging.getLogger("cap_harness.g2s2.serve")


def harness_commit() -> str:
    try:
        root = Path(__file__).resolve().parents[3]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "src"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class S2Worker:
    def __init__(
        self,
        *,
        out_dir: Path,
        program: Path | None = None,
        programs: Path | None = None,
        host: str = "127.0.0.1",
        port: int = 9601,
        sam3_url: str | None = None,
        max_episodes: int | None = None,
        idle_exit_s: float | None = None,
        save_first_requests: bool = False,
        keepalive_s: float | None = None,
        settle_rows: int = 8,
        hold_waist: bool = True,
        keepalive_runtime: dict[str, float] | None = None,
    ) -> None:
        if program is None and programs is None:
            raise ValueError("give --program or --programs")
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.program = program
        self.programs = programs
        self.max_episodes = max_episodes
        self.idle_exit_s = idle_exit_s
        self.settle_rows = settle_rows
        self.hold_waist = hold_waist
        self.sam3 = Sam3Client(sam3_url) if sam3_url else Sam3Client()
        from cap_harness.g2s2.bridge import DEFAULT_KEEPALIVE_BY_RUNTIME

        by_runtime = dict(DEFAULT_KEEPALIVE_BY_RUNTIME)
        by_runtime.update(keepalive_runtime or {})
        self.bridge = S2Bridge(
            host, port, keepalive_s=keepalive_s, keepalive_by_runtime=by_runtime,
            save_first_requests=(out_dir / "first_requests") if save_first_requests else None,
            log_path=out_dir / "bridge.jsonl",
        )
        self.served = 0
        self._stop = threading.Event()
        from cap_harness.g2s2.episode import _dev_fast_end_rows

        self.build = {"harness": PACKAGE_VERSION, "commit": harness_commit(), "hold_waist": hold_waist,
                      "dev_fast_end_rows": _dev_fast_end_rows()}

    def stop(self) -> None:
        self._stop.set()

    def program_for(self, task_name: str | None) -> Path:
        if self.program is not None:
            return self.program
        assert self.programs is not None
        if task_name:
            candidate = self.programs / f"{task_name}.py"
            if candidate.is_file():
                return candidate
        return self.programs / "default.py"

    def run(self) -> int:
        self.bridge.start()
        self._append({"event": "worker_started", "port": self.bridge.bound_port, "pid": os.getpid(), **self.build})
        idle_since = time.time()
        carry = None
        try:
            while not self._stop.is_set():
                if self.max_episodes is not None and self.served >= self.max_episodes:
                    break
                event = carry or self.bridge.next_event(timeout=1.0)
                carry = None
                if event is None:
                    if self.bridge.connected:
                        idle_since = time.time()
                    elif self.idle_exit_s is not None and time.time() - idle_since > self.idle_exit_s:
                        break
                    continue
                if event.kind != "request":
                    continue
                carry = self._serve_episode(event)
                idle_since = time.time()
        finally:
            self._append({"event": "worker_stopped", "served": self.served})
            self.bridge.close()
        return self.served

    def _serve_episode(self, event: Any) -> Any:
        first = event.request
        task = resolve_task(first.embodiment.robot_type, first.embodiment.ee_type, first.task_name)
        index = self.served
        name = task.name if task is not None else f"unlisted_{first.task_name}"
        folder = self.out_dir / f"ep_{index:04d}_{name}"
        folder.mkdir(parents=True, exist_ok=True)
        episode = Episode(self.bridge, first, task, log_dir=folder, connection=event.connection, hold_waist=self.hold_waist)
        api = S2Api(episode, sam3=self.sam3, settle_rows=self.settle_rows, limits=Limits())
        program_path = self.program_for(task.name if task else None)
        code = program_path.read_text(encoding="utf-8") if program_path.is_file() else ""
        report: dict[str, Any] = {}
        logger.info("episode %d: %s (%s, %s) program %s", index, name, first.embodiment.key, episode.runtime, program_path.name)
        started = time.time()
        result = run_program(code, api, report) if code else None
        try:
            outcome = episode.end()
        except Exception as exc:
            logger.exception("episode end failed")
            outcome = episode.outcome
            outcome.end_reason = f"end_failed: {exc}"
        self.served += 1
        record = {
            "event": "episode",
            "index": index,
            "task": name,
            "task_resolved": task is not None,
            "program": str(program_path),
            "program_md5": hashlib.md5(code.encode()).hexdigest() if code else None,
            "program_result": None if result is None else result.as_dict(),
            "episode": episode.summary(),
            "api": api.summary(),
            "report": report,
            "wall_s": round(time.time() - started, 3),
            "build": self.build,
        }
        (folder / "episode.json").write_text(json.dumps(record, indent=1, default=_json_default) + "\n", encoding="utf-8")
        self._append({k: record[k] for k in ("event", "index", "task", "program_md5", "wall_s")} | {
            "program_ok": None if result is None else result.ok,
            "error": None if result is None else result.error,
            "end_reason": episode.outcome.end_reason,
            "dispatches": None if episode.budget is None else len(episode.budget.dispatches),
            "phase_N": None if episode.budget is None else episode.budget.phase_dispatches,
            "rows": None if episode.budget is None else episode.budget.total_rows,
            "folder": str(folder),
        })
        return outcome.carry_over

    def _append(self, record: dict[str, Any]) -> None:
        record.setdefault("at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with (self.out_dir / "episodes.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--programs", type=Path)
    parser.add_argument("--sam3-url")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--idle-exit-s", type=float)
    parser.add_argument("--keepalive-s", type=float)
    parser.add_argument("--keepalive-runtime", action="append", default=[], metavar="RUNTIME=SECONDS",
                        help="override a runtime's keepalive (defaults: rubik 50, newton 600); platform jobs answer within 120 s")
    parser.add_argument("--save-first-requests", action="store_true")
    parser.add_argument("--settle-rows", type=int, default=8)
    parser.add_argument("--no-hold-waist", action="store_true", help="leave the waist uncommanded outside WBC tasks")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker = S2Worker(
        out_dir=args.out, program=args.program, programs=args.programs, host=args.host, port=args.port,
        sam3_url=args.sam3_url, max_episodes=args.max_episodes, idle_exit_s=args.idle_exit_s,
        save_first_requests=args.save_first_requests, keepalive_s=args.keepalive_s, settle_rows=args.settle_rows,
        hold_waist=not args.no_hold_waist,
        keepalive_runtime={kv.split("=", 1)[0]: float(kv.split("=", 1)[1]) for kv in args.keepalive_runtime},
    )
    signal.signal(signal.SIGTERM, lambda *_: worker.stop())
    served = worker.run()
    logger.info("served %d episodes", served)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
