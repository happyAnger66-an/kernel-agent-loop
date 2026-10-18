from dataclasses import dataclass

import numpy as np

from kernel_agent_loop.core.types import Shape


@dataclass(frozen=True)
class CorrectnessReport:
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


def check_correctness(spec, run, shapes: list[Shape], *, seed: int) -> CorrectnessReport:
    failures = []
    for index, shape in enumerate(shapes):
        for round_index in range(2):
            round_seed = seed + index * 10007 + round_index + 1
            try:
                failure = _check_once(spec, run, shape, round_seed)
            except Exception as exc:
                failure = f"{shape.label}: error: {exc}"
            if failure:
                failures.append(failure)
    return CorrectnessReport(tuple(failures))


def _check_once(spec, run, shape: Shape, seed: int) -> str | None:
    inputs = spec.make_inputs(shape, seed=seed)
    original = {name: value.copy() for name, value in inputs.items()}
    outputs = {
        name: np.full(out_shape, np.nan, dtype=dtype)
        for name, (out_shape, dtype) in spec.output_specs(shape).items()
    }
    run(inputs, outputs, shape)
    if any(not np.array_equal(inputs[name], original[name]) for name in inputs):
        return f"{shape.label}: input_modified"
    expected = spec.reference(original)
    for name, actual in outputs.items():
        rtol, atol = spec.tolerance(actual.dtype)
        if not np.allclose(actual, expected[name], rtol=rtol, atol=atol, equal_nan=False):
            return f"{shape.label}: mismatch"
    return None
