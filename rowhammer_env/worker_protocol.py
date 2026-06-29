from __future__ import annotations

import json
import pathlib
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkerRequest:
    kind: str
    request_id: str
    args: tuple[str, ...]

    def line(self) -> str:
        return " ".join((self.kind, self.request_id, *self.args))


class WorkerClient:
    def __init__(self, worker_path: pathlib.Path, config_path: pathlib.Path) -> None:
        self.worker_path = worker_path
        self.config_path = config_path
        if not worker_path.is_file() or not config_path.is_file():
            raise FileNotFoundError("Phase 2 worker is not built")
        self.proc = subprocess.Popen(
            [str(worker_path), str(config_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def call(self, request: WorkerRequest) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("worker pipes are closed")
        self.proc.stdin.write(request.line() + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"worker exited without response: {stderr.strip()}")
        return json.loads(line)

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.call(WorkerRequest("QUIT", "quit", ()))
            except Exception:
                self.proc.kill()
        self.proc.wait(timeout=2)
        for pipe in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    def __enter__(self) -> "WorkerClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
