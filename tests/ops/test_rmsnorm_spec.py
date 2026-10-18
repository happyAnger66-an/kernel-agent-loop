"""模块: ops/rmsnorm 的 OpSpec
功能: 公开 shape、输入生成、输出规格，以及 fp64 参考实现和 fp16 容差。
"""

import numpy as np
import pytest

from kernel_agent_loop.core.opspec import load_opspec


@pytest.fixture
def spec(repo_root):
    return load_opspec(repo_root / "ops", "rmsnorm")


def rmsnorm_fp64(x, w, eps):
    x = x.astype(np.float64)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w.astype(np.float64)


# 用例: 公开 shape 非空且 label 不重复
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_public_shapes_are_nonempty_with_unique_labels(spec):
    shapes = spec.public_shapes()
    assert shapes
    assert len({s.label for s in shapes}) == len(shapes)


# 用例: 生成的输入形状和 dtype 与 shape 声明一致
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_inputs_match_shape_and_dtype(spec):
    shape = spec.public_shapes()[0]
    inputs = spec.make_inputs(shape, seed=0)
    assert inputs["x"].shape == (shape.dims["rows"], shape.dims["hidden"])
    assert inputs["weight"].shape == (shape.dims["hidden"],)
    assert inputs["x"].dtype == np.float16 and inputs["weight"].dtype == np.float16


# 用例: 同一种子生成的输入相同，不同种子生成的输入不同
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_inputs_are_deterministic_per_seed_and_differ_across_seeds(spec):
    shape = spec.public_shapes()[0]
    a, b, c = (spec.make_inputs(shape, seed=s)["x"] for s in (1, 1, 2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# 用例: 输出 y 的形状和 dtype 与输入 x 一致
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_output_spec_matches_input_shape(spec):
    shape = spec.public_shapes()[0]
    out_shape, dtype = spec.output_specs(shape)["y"]
    assert out_shape == (shape.dims["rows"], shape.dims["hidden"])
    assert dtype == np.float16


# 用例: 参考实现与 fp64 的 RMSNorm 公式一致，误差在 fp16 容差内
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_reference_matches_fp64_formula(spec):
    for shape in spec.public_shapes():
        inputs = spec.make_inputs(shape, seed=3)
        expected = rmsnorm_fp64(inputs["x"], inputs["weight"], spec.eps)
        rtol, atol = spec.tolerance(np.float16)
        np.testing.assert_allclose(spec.reference(inputs)["y"], expected, rtol=rtol, atol=atol)


# 用例: fp16 容差足够严，漏乘 weight 的结果过不了
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_fp16_tolerance_is_tight_enough_to_catch_a_dropped_weight(spec):
    shape = spec.public_shapes()[0]
    inputs = spec.make_inputs(shape, seed=4)
    wrong = rmsnorm_fp64(inputs["x"], np.ones_like(inputs["weight"]), spec.eps)
    rtol, atol = spec.tolerance(np.float16)
    assert not np.allclose(spec.reference(inputs)["y"], wrong, rtol=rtol, atol=atol)
