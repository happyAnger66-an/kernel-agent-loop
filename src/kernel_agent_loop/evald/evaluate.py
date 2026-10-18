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
) -> Evaluation:
    """Compile the working-tree impl, check it, time it against campaign.best, and log the verdict.

    Does not accept or revert. The caller does that from decision.verdict.
    A pinned best that cannot be built is treated as no incumbent.
    """
    if backend is None:
        if campaign.backend != "cuda":
            raise ValueError(f"no backend registered for {campaign.backend}")
        backend = CudaBackend()
    if gate is None:
        gate = AcceptanceGate(env_guard=EnvGuard(max_temp_drift_c=5.0), min_gain_pct=1.0)
    if env_probe is None:
        env_probe = read_gpu_env
    if shapes is None:
        shapes = spec.public_shapes()

    source_dir = campaign.repo / "ops" / campaign.op / "impls" / campaign.backend
    env_before = env_probe()
    try:
        run = backend.load(backend.build(source_dir))
    except BuildError as exc:
        return _finish(
            campaign, log, lever, hypothesis, commit, env_before, env_probe(),
            compile_ok=False, correct=False, per_shape={}, symptoms=(str(exc),), gate=gate,
        )

    incumbent_run = None
    if commit != campaign.best:
        incumbent_run = _load_incumbent(campaign, backend)

    report = check_correctness(spec, run, shapes, seed=seed)
    if not report.ok:
        return _finish(
            campaign, log, lever, hypothesis, commit, env_before, env_probe(),
            compile_ok=True, correct=False, per_shape={}, symptoms=report.failures, gate=gate,
        )

    per_shape = {
        shape.label: _time_shape(spec, run, incumbent_run, shape, rounds=rounds, reps=reps, seed=seed)
        for shape in shapes
    }
    return _finish(
        campaign, log, lever, hypothesis, commit, env_before, env_probe(),
        compile_ok=True, correct=True, per_shape=per_shape, symptoms=(), gate=gate,
    )


def _finish(
    campaign: Campaign,
    log: ExperimentLog,
    lever: str,
    hypothesis: str,
    commit: str,
    env_before: EnvSnapshot,
    env_after: EnvSnapshot,
    *,
    compile_ok: bool,
    correct: bool,
    per_shape: dict[str, ShapePerf],
    symptoms: tuple[str, ...],
    gate: AcceptanceGate,
) -> Evaluation:
    result = EvalResult(
        commit=commit,
        compile_ok=compile_ok,
        correct=correct,
        env_before=env_before,
        env_after=env_after,
        per_shape=per_shape,
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
                for label, perf in per_shape.items()
            },
        )
    )
    return Evaluation(result, decision)


def _load_incumbent(campaign: Campaign, backend):
    directory = _materialize_impl(campaign.repo, campaign.best, f"ops/{campaign.op}/impls/{campaign.backend}")
    if directory is None:
        return None
    try:
        return backend.load(backend.build(directory))
    except Exception:
        # The pinned best may be a placeholder that does not build. That is "no incumbent",
        # not a failure of the candidate being evaluated.
        return None


def _materialize_impl(repo: Path, sha: str, impl_rel: str) -> Path | None:
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", sha, "--", impl_rel],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = [line for line in listed.stdout.splitlines() if line]
    if not paths:
        return None
    out = Path(tempfile.mkdtemp(prefix="kal-incumbent-"))
    for rel in paths:
        blob = subprocess.run(
            ["git", "show", f"{sha}:{rel}"],
            cwd=repo,
            check=True,
            capture_output=True,
        )
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
