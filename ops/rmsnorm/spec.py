import numpy as np

from kernel_agent_loop.core.types import Shape


class RmsNormSpec:
    eps = 1e-6

    def public_shapes(self) -> list[Shape]:
        return [
            Shape("r1-h4096", {"rows": 1, "hidden": 4096}),
            Shape("r8-h4096", {"rows": 8, "hidden": 4096}),
            Shape("r32-h1024", {"rows": 32, "hidden": 1024}),
        ]

    def make_inputs(self, shape: Shape, seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        rows, hidden = shape.dims["rows"], shape.dims["hidden"]
        return {
            "x": rng.standard_normal((rows, hidden)).astype(np.float16),
            "weight": rng.uniform(0.25, 1.75, size=hidden).astype(np.float16),
        }

    def output_specs(self, shape: Shape) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
        return {"y": ((shape.dims["rows"], shape.dims["hidden"]), np.float16)}

    def reference(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        x = inputs["x"].astype(np.float64)
        weight = inputs["weight"].astype(np.float64)
        y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.eps) * weight
        return {"y": y.astype(np.float16)}

    def tolerance(self, dtype: np.dtype) -> tuple[float, float]:
        if np.dtype(dtype) == np.float16:
            return 1e-2, 1e-2
        raise ValueError(f"no tolerance for {dtype}")


SPEC = RmsNormSpec()
