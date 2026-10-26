"""模块: kernel_agent_loop.evald.diagnose
功能: ncu 指标按规则归纳成症状；每条症状和理想值比较、带证据，stall/occupancy 类只在还有 headroom 时报。
"""

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from kernel_agent_loop.core.opspec import load_opspec
from kernel_agent_loop.evald.diagnose import diagnose_shape, summarize, symptom_codes

# Thor 上仓库 baseline（标量读、单 block）在 r1-h4096 的实测指标
BASELINE_R1 = {
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 768.0,
    "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum": 384.0,
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": 258.0,
    "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum": 128.0,
    "lts__t_sectors_srcunit_tex_op_read.sum": 512.0,
    "lts__t_sectors_srcunit_tex_op_write.sum": 256.0,
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": 0.33,
    "launch__grid_size": 1.0,
    "launch__waves_per_multiprocessor": 0.01,
    "sm__maximum_warps_per_active_cycle_pct": 100.0,
    "launch__registers_per_thread": 18.0,
    "device__attribute_multiprocessor_count": 20.0,
    "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio": 24351.0,
    "smsp__average_warp_latency_issue_stalled_barrier.ratio": 186.0,
    "smsp__average_warp_latency_issue_stalled_wait.ratio": 1433.0,
}
R1_BYTES = dict(read_bytes=16384, written_bytes=8192)


def codes(findings):
    return {f.code for f in findings}


# 用例: baseline 的 r1 报“访存太窄、GPU 没填满、等访存”，不报不合并和 L2 重取
# 参数: 无
def test_baseline_r1_symptoms():
    found = codes(diagnose_shape("r1", BASELINE_R1, headroom=1.48, **R1_BYTES))
    assert found == {"narrow-global-access", "underfilled-gpu", "memory-latency-bound"}


# 用例: 已贴近下限的 shape 不报 stall/填充类症状，只报 near-floor；确定的浪费（访存太窄）照报
# 参数: 无
def test_near_floor_suppresses_stall_and_fill_symptoms():
    found = codes(diagnose_shape("r1", BASELINE_R1, headroom=1.02, **R1_BYTES))
    assert found == {"narrow-global-access", "near-floor"}


# 用例: 每请求超过 16 个扇区判为不合并，不当成“太窄”
# 参数: 无
def test_more_than_16_sectors_per_request_is_uncoalesced():
    m = {**BASELINE_R1, "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": 384.0 * 30}
    load = [f for f in diagnose_shape("r1", m, headroom=1.02, **R1_BYTES) if "load" in f.evidence]
    assert [f.code for f in load] == ["uncoalesced-global-access"]


# 用例: L2 读流量超过理想字节 1.25 倍且还有 headroom 时判为重取；贴近下限时（如 weight 按 SM 广播）不报
# 参数: 无
def test_l2_read_traffic_above_ideal_is_refetch_only_with_headroom():
    m = {**BASELINE_R1, "lts__t_sectors_srcunit_tex_op_read.sum": 1024.0}
    found = diagnose_shape("r1", m, headroom=1.5, **R1_BYTES)
    assert any(f.code == "l2-refetch" and "2.00x" in f.evidence for f in found)
    assert "l2-refetch" not in codes(diagnose_shape("r1", m, headroom=1.03, **R1_BYTES))


# 用例: 4.27 波时报尾波，并估出空闲比例
# 参数: 无
def test_partial_last_wave_is_reported():
    m = {"launch__waves_per_multiprocessor": 4.27, "launch__grid_size": 512.0,
         "device__attribute_multiprocessor_count": 20.0}
    found = diagnose_shape("r512", m, headroom=1.18, read_bytes=1, written_bytes=1)
    assert [f.code for f in found] == ["tail-wave"]
    assert "27%" in found[0].evidence


# 用例: 理论 occupancy 低于 50% 时报出限制因素
# 参数: 无
def test_low_theoretical_occupancy_names_limiter():
    m = {"sm__maximum_warps_per_active_cycle_pct": 25.0, "launch__waves_per_multiprocessor": 3.0,
         "launch__occupancy_limit_registers": 2.0, "launch__occupancy_limit_warps": 6.0,
         "launch__registers_per_thread": 128.0}
    found = diagnose_shape("s", m, headroom=1.5, read_bytes=1, written_bytes=1)
    assert [f.code for f in found] == ["occupancy-limited"]
    assert "registers" in found[0].evidence and "128" in found[0].evidence


# 用例: barrier stall 占比超过 20% 时报
# 参数: 无
def test_barrier_heavy_stalls_are_reported():
    m = {"smsp__average_warp_latency_issue_stalled_barrier.ratio": 400.0,
         "smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio": 600.0,
         "lts__throughput.avg.pct_of_peak_sustained_elapsed": 80.0}
    found = codes(diagnose_shape("s", m, headroom=1.5, read_bytes=1, written_bytes=1))
    assert found == {"barrier-stall"}


# 用例: 同一症状跨 shape 合成一行，带每个 shape 的证据、手段和依据
# 参数: 无
def test_summarize_groups_by_code_with_levers_and_basis():
    findings = diagnose_shape("r1", BASELINE_R1, headroom=1.48, **R1_BYTES)
    findings += diagnose_shape("r8", BASELINE_R1, headroom=1.30, **R1_BYTES)
    lines = summarize(findings)
    narrow = [line for line in lines if line.startswith("[narrow-global-access]")]
    assert len(narrow) == 1
    assert "r1:" in narrow[0] and "r8:" in narrow[0]
    assert "Try: vector-16b (" in narrow[0] and "Basis:" in narrow[0]


# 用例: 只认 [已知规则码] 开头的行，失败原因等其他文本忽略
# 参数: 无
def test_symptom_codes_parses_rule_lines_only():
    lines = ("[underfilled-gpu] r1: 1 blocks", "[made-up] x", "mismatch r1: max_abs=1", "profile unavailable: x")
    assert symptom_codes(lines) == {"underfilled-gpu"}


IMAGE = "adas-img.nioint.com/r-lab/blackwell-aarch64:latest"


def _image_available() -> bool:
    if shutil.which("docker") is None or shutil.which("nvcc") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode == 0


@pytest.mark.thor
@pytest.mark.skipif(not _image_available(), reason="docker, nvcc or the ncu image is missing")
# 用例: 真机标注样本：标量 baseline 报访存太窄，向量化版不报，分块跨步版报不合并；单行 shape 都报 GPU 没填满
# 参数: repo_root 提供实现、样本 kernel 与契约
def test_rules_on_labeled_kernels(repo_root):
    pytest.importorskip("cuda")
    from kernel_agent_loop.backends.cuda import CudaBackend
    from kernel_agent_loop.evald.floor import shape_bytes
    from kernel_agent_loop.evald.profile import ContainerProfiler

    spec = load_opspec(repo_root / "ops", "rmsnorm")
    shapes = [s for s in spec.public_shapes() if s.label in ("r1-h4096", "r512-h4096")]
    impl = repo_root / "ops/rmsnorm/impls/cuda"
    samples = {
        "baseline": impl / "kernel.cu",
        "vec": repo_root / "tests/evald/rmsnorm_vec.cu",
        "strided": repo_root / "tests/evald/rmsnorm_strided.cu",
    }
    backend = CudaBackend()
    found = {}
    for name, source in samples.items():
        work = Path(tempfile.mkdtemp(prefix=f"kal-diag-{name}-"))
        shutil.copy2(source, work / "kernel.cu")
        shutil.copy2(impl / "launch.py", work / "launch.py")
        artifact = backend.build(work)
        kernel = backend.load(artifact)
        try:
            items = [(s, kernel.launch_config(s)) for s in shapes]
        finally:
            kernel.close()
        profile = ContainerProfiler(image=IMAGE)(artifact.cubin_path, spec, items)
        for s in shapes:
            read, written = shape_bytes(spec, s)
            found[name, s.label] = codes(diagnose_shape(s.label, profile[s.label], headroom=None,
                                                        read_bytes=read, written_bytes=written))
    for s in shapes:
        assert "narrow-global-access" in found["baseline", s.label]
        assert "uncoalesced-global-access" not in found["baseline", s.label]
        assert not found["vec", s.label] & {"narrow-global-access", "uncoalesced-global-access"}
        assert "uncoalesced-global-access" in found["strided", s.label]
        assert "l2-refetch" not in found["vec", s.label]
    for name in samples:
        assert "underfilled-gpu" in found[name, "r1-h4096"]
        assert "underfilled-gpu" not in found[name, "r512-h4096"]
