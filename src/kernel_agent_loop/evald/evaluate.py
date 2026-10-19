import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kernel_agent_loop.backends.cuda import BuildError, CudaBackend
from kernel_agent_loop.core.campaign import Campaign
from kernel_agent_loop.core.experiment_log import Attempt, ExperimentLog
from kernel_agent_loop.core.gate import AcceptanceGate, Decision
from kernel_agent_loop.core.types import EnvSnapshot, EvalResult, Shape, ShapePerf
from kernel_agent_loop.evald.correctness import check_correctness
from kernel_agent_loop.evald.env_guard import EnvGuard, read_gpu_env
from kernel_agent_loop.evald.runner import SubprocessKernelRunner, materialize_artifact_dir
from kernel_agent_loop.evald.timing import interleaved_bench


@dataclass(frozen=True)
class Evaluation:
    result: EvalResult
    decision: Decision


def evaluate(
    campaign: Campaign,
    *,
    spec,
    log: ExperimentLog,
    lever: str,
    hypothesis: str,
    commit: str,
    shapes: list[Shape] | None = None,
    backend=None,
    gate: AcceptanceGate | None = None,
    env_probe=None,
    rounds: int = 5,
    reps: int = 3,
    seed: int = 0,
    isolate: bool = False,
    run_timeout: float = 120.0,
    worker_args: list[str] | None = None,
) -> Evaluation:
    """Compile the working-tree impl, check it, time it against campaign.best, and log the verdict.

    Does not accept or revert. The caller does that from decision.verdict.
    A pinned best that cannot be built is treated as no incumbent.
    """
    if gate is None:
        gate = AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0)
    result, symptoms = measure(
        repo=campaign.repo,
        op=campaign.op,
        backend_name=campaign.backend,
        commit=commit,
        best=campaign.best,
        spec=spec,
        shapes=shapes,
        backend=backend,
        env_probe=env_probe,
        rounds=rounds,
        reps=reps,
        seed=seed,
        isolate=isolate,
        run_timeout=run_timeout,
        worker_args=worker_args,
    )
    decision = gate.decide(result)
    log.record(
        Attempt(
            campaign=campaign.campaign_id,
            commit=commit,
            parent=campaign.best,
            lever=lever,
            hypothesis=hypothesis,
            verdict=decision.verdict.value,
            symptoms=symptoms,
            per_shape={
                label: {
                    "candidate_us": perf.candidate_us,
                    "incumbent_us": perf.incumbent_us,
                    "noise_pct": perf.noise_pct,
                }
                for label, perf in result.per_shape.items()
            },
        )
    )
    return Evaluation(result, decision)


def measure(
    *,
    repo: Path,
    op: str,
    backend_name: str,
    commit: str,
    best: str,
    spec,
    shapes: list[Shape] | None = None,
    backend=None,
    env_probe=None,
    rounds: int = 5,
    reps: int = 3,
    seed: int = 0,
    isolate: bool = False,
    run_timeout: float = 120.0,
    worker_args: list[str] | None = None,
) -> tuple[EvalResult, tuple[str, ...]]:
    """Compile, check correctness, interleaved bench, env snapshots. No gate and no experiment log."""
    if backend is None:
        if backend_name != "cuda":
            raise ValueError(f"no backend registered for {backend_name}")
        backend = CudaBackend()
    if env_probe is None:
        env_probe = read_gpu_env
    if shapes is None:
        shapes = spec.public_shapes()

    source_dir = Path(repo) / "ops" / op / "impls" / backend_name
    env_before = env_probe()
    try:
        artifact = backend.build(source_dir)
    except BuildError as exc:
        return (
            _result(commit, env_before, env_probe(), compile_ok=False, correct=False, per_shape={}),
            (str(exc),),
        )

    candidate = _bind_run(artifact, backend, isolate=isolate, run_timeout=run_timeout, worker_args=worker_args)
    incumbent = None
    try:
        if commit != best:
            incumbent = _load_incumbent(
                repo, best, op, backend_name, backend,
                isolate=isolate, run_timeout=run_timeout, worker_args=worker_args,
            )
        report = check_correctness(spec, candidate, shapes, seed=seed)
        if not report.ok:
            return (
                _result(commit, env_before, env_probe(), compile_ok=True, correct=False, per_shape={}),
                report.failures,
            )
        per_shape = {
            shape.label: _time_shape(spec, candidate, incumbent, shape, rounds=rounds, reps=reps, seed=seed)
            for shape in shapes
        }
        return (
            _result(commit, env_before, env_probe(), compile_ok=True, correct=True, per_shape=per_shape),
            (),
        )
    finally:
        _close_run(candidate)
        if incumbent is not None:
            _close_run(incumbent)


def _result(
    commit: str,
    env_before: EnvSnapshot,
    env_after: EnvSnapshot,
    *,
    compile_ok: bool,
    correct: bool,
    per_shape: dict[str, ShapePerf],
) -> EvalResult:
    return EvalResult(
        commit=commit,
        compile_ok=compile_ok,
        correct=correct,
        env_before=env_before,
        env_after=env_after,
        per_shape=per_shape,
    )


def _bind_run(artifact, backend, *, isolate: bool, run_timeout: float, worker_args: list[str] | None):
    if not isolate:
        return backend.load(artifact)
    dest = Path(tempfile.mkdtemp(prefix="kal-runner-"))
    materialize_artifact_dir(artifact, dest)
    return SubprocessKernelRunner(dest, timeout=run_timeout, worker_args=worker_args)


def _close_run(run) -> None:
    close = getattr(run, "close", None)
    if callable(close):
        close()


def _load_incumbent(
    repo: Path,
    sha: str,
    op: str,
    backend_name: str,
    backend,
    *,
    isolate: bool,
    run_timeout: float,
    worker_args: list[str] | None,
):
    directory = _materialize_impl(repo, sha, f"ops/{op}/impls/{backend_name}")
    if directory is None:
        return None
    try:
        artifact = backend.build(directory)
    except Exception:
        # The pinned best may be a placeholder that does not build. That is "no incumbent",
        # not a failure of the candidate being evaluated.
        return None
    return _bind_run(artifact, backend, isolate=isolate, run_timeout=run_timeout, worker_args=worker_args)


def _materialize_impl(repo: Path, sha: str, impl_rel: str) -> Path | None:
    if not (Path(repo) / ".git").exists():
        return None
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", sha, "--", impl_rel],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        return None
    paths = [line for line in listed.stdout.splitlines() if line]
    if not paths:
        return None
    out = Path(tempfile.mkdtemp(prefix="kal-incumbent-"))
    for rel in paths:
        blob = subprocess.run(
            ["git", "show", f"{sha}:{rel}"],
            cwd=repo,
            capture_output=True,
        )
        if blob.returncode != 0:
            return None
        dest = out / Path(rel).relative_to(impl_rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob.stdout)
    return out


def _time_shape(spec, run, incumbent_run, shape: Shape, *, rounds: int, reps: int, seed: int) -> ShapePerf:
    inputs = spec.make_inputs(shape, seed=seed)
    outputs = {
        name: np.empty(out_shape, dtype=dtype)
        for name, (out_shape, dtype) in spec.output_specs(shape).items()
    }

    def timed(fn):
        def once():
            started = time.perf_counter()
            fn(inputs, outputs, shape)
            return (time.perf_counter() - started) * 1e6
        return once

    incumbent = timed(incumbent_run) if incumbent_run is not None else None
    return interleaved_bench(timed(run), incumbent, rounds=rounds, reps=reps)
