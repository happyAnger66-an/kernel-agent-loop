"""Thor-side evald entry: flock, overlay candidate bundle, measure, write EvalResult."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import uuid
from pathlib import Path

from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.core.types import EvalResult
from kernel_agent_loop.evald.bundle import BundleError, overlay_bundle
from kernel_agent_loop.evald.config import EvaldConfig
from kernel_agent_loop.evald.env_guard import read_gpu_env
from kernel_agent_loop.evald.evaluate import measure
from kernel_agent_loop.evald.flockutil import exclusive_lock
from kernel_agent_loop.evald.result_codec import eval_result_to_dict
from kernel_agent_loop.evald.thor_env import prepare_thor_session


def evald_run(
    bundle: bytes,
    config: EvaldConfig,
    *,
    env_probe=None,
    backend=None,
    worker_args: list[str] | None = None,
    job_id: str | None = None,
) -> EvalResult:
    """Hold the GPU lock, overlay the bundle onto a work tree, and return EvalResult."""
    if config.prepare_env:
        prepare_thor_session()
    if env_probe is None:
        env_probe = read_gpu_env

    meta = _read_meta(bundle)
    op = meta["op"]
    backend_name = meta["backend"]
    job_id = job_id or uuid.uuid4().hex
    work = Path(config.work_root) / job_id
    if work.exists():
        shutil.rmtree(work)

    with exclusive_lock(config.lock_path):
        try:
            overlay_bundle(bundle, Path(config.trusted), work, op=op, backend=backend_name)
        except BundleError:
            if work.exists():
                shutil.rmtree(work)
            raise

        ops_root = Path(config.ops_root) if config.ops_root is not None else work / "ops"
        spec = load_opspec(ops_root, op)
        shapes = list(spec.public_shapes()) + list(config.hidden_shapes)
        result, _ = measure(
            repo=work,
            op=op,
            backend_name=backend_name,
            commit=meta["commit"],
            best=meta.get("parent", meta["commit"]),
            spec=spec,
            shapes=shapes,
            backend=backend,
            env_probe=env_probe,
            rounds=config.rounds,
            reps=config.reps,
            isolate=config.isolate,
            run_timeout=config.run_timeout,
            worker_args=worker_args,
        )
        return result


def evald_run_path(
    bundle_path: Path,
    config: EvaldConfig,
    *,
    result_path: Path | None = None,
    env_probe=None,
    backend=None,
    worker_args: list[str] | None = None,
) -> EvalResult:
    result = evald_run(
        Path(bundle_path).read_bytes(),
        config,
        env_probe=env_probe,
        backend=backend,
        worker_args=worker_args,
    )
    if result_path is not None:
        Path(result_path).write_text(json.dumps(eval_result_to_dict(result), indent=2) + "\n")
    return result


def _read_meta(bundle: bytes) -> dict:
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as tar:
        member = tar.getmember("meta.json")
        return json.loads(tar.extractfile(member).read())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one kernel-agent-loop eval job under flock")
    parser.add_argument("--config", required=True, type=Path, help="EvaldConfig JSON path")
    parser.add_argument("--bundle", required=True, type=Path, help="Candidate tar.gz path")
    parser.add_argument("--result", required=True, type=Path, help="Where to write EvalResult JSON")
    parser.add_argument("--fake-worker", action="store_true", help="Use fake runner worker (tests)")
    args = parser.parse_args(argv)

    config = EvaldConfig.load(args.config)
    worker_args = ["--fake"] if args.fake_worker else None
    evald_run_path(args.bundle, config, result_path=args.result, worker_args=worker_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
