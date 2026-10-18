"""模块: kernel_agent_loop.evald.correctness
功能: 正确性检查拦住复用输出、空跑、改写输入、只对公开 shape 正确和候选异常。
"""

import numpy as np
import pytest

from kernel_agent_loop.core.opspec import Shape, load_opspec
from kernel_agent_loop.evald.correctness import check_correctness

HIDDEN = [Shape("hidden-r3-h1000", {"rows": 3, "hidden": 1000})]


@pytest.fixture
def spec(repo_root):
    return load_opspec(repo_root / "ops", "rmsnorm")


def rmsnorm(spec, inputs):
    x = inputs["x"].astype(np.float32)
    y = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + spec.eps) * inputs["weight"].astype(np.float32)
    return y.astype(np.float16)


def honest(spec):
    def run(inputs, outputs, shape):
        outputs["y"][...] = rmsnorm(spec, inputs)
    return run


def stale(spec):
    cache = {}

    def run(inputs, outputs, shape):
        if shape.label not in cache:
            cache[shape.label] = rmsnorm(spec, inputs)
        outputs["y"][...] = cache[shape.label]
    return run


def noop(spec):
    def run(inputs, outputs, shape):
        pass
    return run


def mutates_input(spec):
    def run(inputs, outputs, shape):
        outputs["y"][...] = rmsnorm(spec, inputs)
        inputs["x"][...] = 0
    return run


def overfit_public(spec):
    public = {s.label for s in spec.public_shapes()}

    def run(inputs, outputs, shape):
        outputs["y"][...] = rmsnorm(spec, inputs) if shape.label in public else 0
    return run


def crashes(spec):
    def run(inputs, outputs, shape):
        raise RuntimeError("boom")
    return run


# 用例: 按公式计算的候选在公开 shape 和隐藏 shape 上都通过
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_honest_candidate_passes_public_and_hidden_shapes(spec):
    report = check_correctness(spec, honest(spec), spec.public_shapes() + HIDDEN, seed=0)
    assert report.ok, report.failures


# 用例: 复用上一次输出会被判 mismatch
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_reusing_a_previous_output_is_caught(spec):
    report = check_correctness(spec, stale(spec), spec.public_shapes(), seed=0)
    assert not report.ok
    assert any("mismatch" in f for f in report.failures)


# 用例: 不写输出、留下预先填入的 NaN 时判失败
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_leaving_poisoned_output_untouched_is_caught(spec):
    report = check_correctness(spec, noop(spec), spec.public_shapes(), seed=0)
    assert not report.ok


# 用例: 结果正确但改写了输入时，按 input_modified 判失败
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_writing_into_inputs_is_caught(spec):
    report = check_correctness(spec, mutates_input(spec), spec.public_shapes(), seed=0)
    assert not report.ok
    assert any("input_modified" in f for f in report.failures)


# 用例: 只对公开 shape 算对的候选，公开集通过、加上隐藏 shape 后失败
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_hidden_shapes_catch_what_public_shapes_miss(spec):
    run = overfit_public(spec)
    assert check_correctness(spec, run, spec.public_shapes(), seed=0).ok
    report = check_correctness(spec, run, spec.public_shapes() + HIDDEN, seed=0)
    assert not report.ok
    assert any("hidden-r3-h1000" in f for f in report.failures)


# 用例: 候选抛异常时记为 error，不让评测进程崩掉
# 参数: spec 是仓库里的 rmsnorm OpSpec
def test_candidate_exception_is_reported_not_raised(spec):
    report = check_correctness(spec, crashes(spec), spec.public_shapes(), seed=0)
    assert not report.ok
    assert any("error" in f for f in report.failures)
