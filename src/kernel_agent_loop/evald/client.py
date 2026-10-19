"""Dev-machine EvalClient: local or SSH-sync submit of a candidate bundle."""

from __future__ import annotations

import json
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from kernel_agent_loop.core.types import EvalResult
from kernel_agent_loop.evald.config import EvaldConfig
from kernel_agent_loop.evald.result_codec import eval_result_from_dict
from kernel_agent_loop.evald.serve import evald_run


@dataclass
class EvalClient:
    """Synchronous evaluator client. Gate and experiment log stay on the caller side."""

    mode: str = "local"  # "local" | "ssh"
    config: EvaldConfig | None = None
    host: str = ""
    user: str = ""
    remote_workdir: str = "/tmp/kal-eval"
    evald_cmd: str = "python -m kernel_agent_loop.evald.serve"
    remote_config: str = ""
    ssh_opts: tuple[str, ...] = ()
    backend: object | None = None
    env_probe: object | None = None
    worker_args: list[str] | None = None

    def submit_sync(self, bundle: bytes) -> EvalResult:
        if self.mode == "local":
            if self.config is None:
                raise ValueError("local EvalClient requires config")
            return evald_run(
                bundle,
                self.config,
                env_probe=self.env_probe,
                backend=self.backend,
                worker_args=self.worker_args,
            )
        if self.mode == "ssh":
            return self._submit_ssh(bundle)
        raise ValueError(f"unknown EvalClient mode {self.mode!r}")

    def _submit_ssh(self, bundle: bytes) -> EvalResult:
        if not self.host or not self.remote_config:
            raise ValueError("ssh EvalClient requires host and remote_config")
        job = uuid.uuid4().hex
        remote_dir = f"{self.remote_workdir.rstrip('/')}/{job}"
        target = self._ssh_target()
        self._run(["ssh", *self.ssh_opts, target, f"mkdir -p {remote_dir}"])
        with tempfile.TemporaryDirectory(prefix="kal-client-") as tmp:
            local_bundle = Path(tmp) / "candidate.tar.gz"
            local_result = Path(tmp) / "result.json"
            local_bundle.write_bytes(bundle)
            self._run(["scp", *self.ssh_opts, str(local_bundle), f"{target}:{remote_dir}/candidate.tar.gz"])
            remote_result = f"{remote_dir}/result.json"
            cmd = (
                f"{self.evald_cmd} --config {self.remote_config} "
                f"--bundle {remote_dir}/candidate.tar.gz --result {remote_result}"
            )
            self._run(["ssh", *self.ssh_opts, target, cmd])
            self._run(["scp", *self.ssh_opts, f"{target}:{remote_result}", str(local_result)])
            data = json.loads(local_result.read_text())
        # Best-effort cleanup; ignore failures.
        subprocess.run(["ssh", *self.ssh_opts, target, f"rm -rf {remote_dir}"], check=False)
        return eval_result_from_dict(data)

    def _ssh_target(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    @staticmethod
    def _run(cmd: list[str]) -> None:
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"command failed ({completed.returncode}): {' '.join(cmd)}\n"
                f"{completed.stderr or completed.stdout}"
            )
