"""Subprocess kernel runner: child loads a built artifact and serves run() over JSONL."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from kernel_agent_loop.core.types import Shape
from kernel_agent_loop.evald.array_codec import decode_array, encode_array


class RunnerError(Exception):
    pass


class SubprocessKernelRunner:
    """Long-lived child that only loads an artifact and executes run requests."""

    def __init__(
        self,
        artifact_dir: Path,
        *,
        incumbent_dir: Path | None = None,
        timeout: float = 120.0,
        worker_args: list[str] | None = None,
    ) -> None:
        self.timeout = timeout
        self._staged: tuple[dict, dict, Shape] | None = None
        self._hello: dict | None = None
        cmd = [sys.executable, "-m", "kernel_agent_loop.evald.runner_worker", str(artifact_dir)]
        if incumbent_dir is not None:
            cmd.extend(["--incumbent", str(incumbent_dir)])
        if worker_args:
            cmd.extend(worker_args)
        env = os.environ.copy()
        src_root = str(Path(__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = src_root if not existing else f"{src_root}{os.pathsep}{existing}"
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )

    def __call__(self, inputs: dict, outputs: dict, shape: Shape) -> None:
        self.run(inputs, outputs, shape)

    def run(self, inputs: dict, outputs: dict, shape: Shape) -> None:
        response = self._request("run", inputs, outputs, shape)
        for name, payload in response["outputs"].items():
            outputs[name][...] = decode_array(payload)

    @property
    def has_incumbent(self) -> bool:
        """Whether the child loaded an incumbent; one that fails to load counts as absent."""
        return bool(self._ready().get("incumbent"))

    def bench(self, inputs: dict, outputs: dict, shape: Shape) -> float:
        """Per-launch microseconds as timed inside the child, excluding this IPC round trip."""
        self._stage(inputs, outputs, shape)
        return float(self._request("bench", None, None, shape)["us"])

    def bench_pair(self, inputs: dict, outputs: dict, shape: Shape) -> tuple[float, float]:
        """(candidate_us, incumbent_us) timed together in the child on the same buffers."""
        self._stage(inputs, outputs, shape)
        response = self._request("bench_pair", None, None, shape)
        return float(response["candidate_us"]), float(response["incumbent_us"])

    def _stage(self, inputs: dict, outputs: dict, shape: Shape) -> None:
        """Ship arrays once per (inputs, outputs, shape); later bench calls reuse them."""
        staged = self._staged
        if staged is None or staged[0] is not inputs or staged[1] is not outputs or staged[2] != shape:
            self._request("stage", inputs, outputs, shape)
            self._staged = (inputs, outputs, shape)

    def _ready(self) -> dict:
        if self._hello is None:
            assert self._proc.stdout is not None
            line = self._proc.stdout.readline()
            try:
                self._hello = json.loads(line) if line else {"ok": False, "error": "runner closed stdout"}
            except json.JSONDecodeError:
                self._hello = {"ok": False, "error": f"invalid runner hello: {line!r}"}
            if not self._hello.get("ok"):
                self._hello["error"] = f"{self._hello.get('error', 'runner failed')}; stderr={self._stderr()}"
        return self._hello

    def _request(self, cmd: str, inputs: dict | None, outputs: dict | None, shape: Shape) -> dict:
        hello = self._ready()
        if not hello.get("ok"):
            raise RunnerError(hello["error"])
        if self._proc.poll() is not None:
            raise RunnerError(f"runner exited early: {self._stderr()}")
        request = {"cmd": cmd, "shape": {"label": shape.label, "dims": shape.dims}}
        if inputs is not None:
            request["inputs"] = {name: encode_array(value) for name, value in inputs.items()}
        if outputs is not None:
            request["outputs"] = {name: encode_array(value) for name, value in outputs.items()}
        assert self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RunnerError(f"runner closed stdout: {self._stderr()}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RunnerError(f"invalid runner response: {line!r}; stderr={self._stderr()}") from exc
        if not response.get("ok"):
            raise RunnerError(response.get("error", "runner failed"))
        return response

    def close(self) -> None:
        if self._proc.poll() is not None:
            return
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self._proc.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            self._proc.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5)

    def _stderr(self) -> str:
        if self._proc.stderr is None:
            return ""
        try:
            return self._proc.stderr.read()
        except Exception:
            return ""

    def __enter__(self) -> "SubprocessKernelRunner":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def materialize_artifact_dir(artifact, dest: Path) -> Path:
    """Copy cubin + launch.py into dest so a child can load without the original tree."""
    dest.mkdir(parents=True, exist_ok=True)
    cubin_dest = dest / "kernel.cubin"
    if Path(artifact.cubin_path).resolve() != cubin_dest.resolve():
        cubin_dest.write_bytes(Path(artifact.cubin_path).read_bytes())
    launch_src = Path(artifact.source_dir) / "launch.py"
    if launch_src.is_file():
        (dest / "launch.py").write_bytes(launch_src.read_bytes())
    kernel_src = Path(artifact.source_dir) / "kernel.cu"
    if kernel_src.is_file():
        (dest / "kernel.cu").write_bytes(kernel_src.read_bytes())
    return dest
