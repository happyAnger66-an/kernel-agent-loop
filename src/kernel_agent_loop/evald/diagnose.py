"""Turn ncu metrics into symptoms: each rule compares a measurement with an ideal and names its evidence.

Ideals come from the op spec (bytes an ideal impl reads and writes) and the device (SM count). Thresholds
follow the matching ncu rule script (sections/*.py) where one exists. Stall and occupancy symptoms only
count when the shape still has headroom over its measured floor: a memory-bound kernel at the floor is
supposed to wait on memory.
"""

from __future__ import annotations

from dataclasses import dataclass

SECTOR_BYTES = 32
MAX_COALESCED_SECTORS = 16  # 32 threads x 16 B, the widest per-thread access
HEADROOM_GATE = 1.15
NEAR_FLOOR = 1.08

STALLS = (
    "long_scoreboard", "short_scoreboard", "barrier", "membar", "mio_throttle", "lg_throttle",
    "wait", "math_pipe_throttle", "not_selected", "no_instruction", "branch_resolving", "dispatch_stall",
)


@dataclass(frozen=True)
class Rule:
    levers: tuple[tuple[str, str], ...]  # (lever id to pass as run_round's lever, description)
    basis: str


RULES = {
    "narrow-global-access": Rule(
        (("vector-16b", "load/store 16 B per thread (uint4 / 8 halves) from 16 B-aligned addresses, "
                        "scalar tail for the rest"),),
        "sectors per request = bytes per thread for contiguous access; ideal 16 (CUDA Best Practices)",
    ),
    "uncoalesced-global-access": Rule(
        (("coalesce-threads", "make consecutive threads touch consecutive 16 B chunks"),
         ("smem-transpose", "stage through shared memory when the natural layout is strided")),
        "more than 16 sectors per warp request cannot come from contiguous access (ncu UncoalescedAccess)",
    ),
    "l2-refetch": Rule(
        (("single-pass-row", "read each input once: keep the row in registers or shared memory between passes"),
         ("block-order-l2", "order blocks so data reused across blocks is still in L2")),
        "L2 sectors from L1 vs the spec's ideal bytes (ncu MemoryCacheAccessPattern)",
    ),
    "underfilled-gpu": Rule(
        (("split-row-blocks", "split each row across several blocks (thread-block cluster / DSMEM or a "
                              "two-pass reduction)"),
         ("warp-per-row", "shorten the per-block dependency chain: one warp per row, warp-shuffle reduction, "
                          "no __syncthreads")),
        "fewer blocks than SMs leaves SMs idle (ncu LaunchStatistics)",
    ),
    "tail-wave": Rule(
        (("rows-per-block-waves", "pick rows per block so the block count is a multiple of SMs x resident "
                                  "blocks per SM"),
         ("persistent-grid", "persistent grid of exactly SMs x blocks per SM with a grid-stride loop over rows")),
        "a partial last wave idles the rest of the SMs (ncu LaunchStatistics)",
    ),
    "occupancy-limited": Rule(
        (("cut-registers", "lower registers per thread (__launch_bounds__, fewer live values) or shared "
                           "memory per block"),
         ("resize-block", "change block size so more blocks fit per SM")),
        "theoretical occupancy and its limiter (ncu TheoreticalOccupancy)",
    ),
    "barrier-stall": Rule(
        (("warp-shuffle-reduce", "replace shared-memory tree reductions with warp-shuffle reductions"),
         ("fewer-syncs", "cut __syncthreads: one warp per row, or a single block-wide sync")),
        "share of warp stall cycles at barriers (ncu CPIStall / WarpStateStats)",
    ),
    "memory-latency-bound": Rule(
        (("more-bytes-in-flight", "more bytes in flight per thread: 16 B accesses, unroll independent loads, "
                                  "prefetch the next row"),
         ("more-warps", "more resident warps per SM")),
        "warps mostly wait on global memory while L2 throughput stays low (ncu CPIStall + SpeedOfLight)",
    ),
    "near-floor": Rule(
        (("skip-shape", "leave this shape alone; spend rounds on shapes with more headroom"),),
        "candidate within 8% of the measured touch-kernel floor (evald.floor)",
    ),
}


@dataclass(frozen=True)
class Finding:
    code: str
    shape: str
    evidence: str


def diagnose_shape(label: str, m: dict[str, float], *, headroom: float | None,
                   read_bytes: int, written_bytes: int) -> list[Finding]:
    findings = []

    def add(code: str, evidence: str) -> None:
        findings.append(Finding(code, label, evidence))

    has_room = headroom is None or headroom > HEADROOM_GATE
    room = "" if headroom is None else f", headroom {headroom:.2f}x"

    for op, name in (("ld", "load"), ("st", "store")):
        sectors = m.get(f"l1tex__t_sectors_pipe_lsu_mem_global_op_{op}.sum")
        requests = m.get(f"l1tex__t_requests_pipe_lsu_mem_global_op_{op}.sum")
        if not sectors or not requests:
            continue
        per_request = sectors / requests
        if per_request > MAX_COALESCED_SECTORS:
            add("uncoalesced-global-access", f"{per_request:.1f} sectors per global {name} request (max 16 if contiguous)")
        elif per_request < 8:
            add("narrow-global-access",
                f"{per_request:.1f} sectors per global {name} request, about {per_request:.0f} B per thread (ideal 16)")

    # Operands broadcast to every block (e.g. RMSNorm weight) are fetched from L2 once per SM, which is
    # cheap; so extra L2 traffic only counts as a symptom when the shape is also far from its floor.
    for op, ideal, name in (("read", read_bytes, "read"), ("write", written_bytes, "written")):
        sectors = m.get(f"lts__t_sectors_srcunit_tex_op_{op}.sum")
        if has_room and sectors and ideal > 0:
            ratio = sectors * SECTOR_BYTES / ideal
            if ratio > 1.25:
                add("l2-refetch", f"L2 {op} traffic {ratio:.2f}x the {ideal} B an ideal impl has {name}{room}")

    sms = m.get("device__attribute_multiprocessor_count")
    grid = m.get("launch__grid_size")
    waves = m.get("launch__waves_per_multiprocessor")
    if has_room and sms and grid is not None and grid < sms:
        add("underfilled-gpu", f"{grid:.0f} blocks for {sms:.0f} SMs{room}")
    if has_room and waves is not None and 1 <= waves < 8:
        last = waves - int(waves)
        if 0 < last <= 0.5:
            idle = (1 - last) / (int(waves) + 1)
            add("tail-wave", f"{waves:.2f} waves, last wave fills {last:.0%} of SMs, ~{idle:.0%} of SM time idle{room}")

    theoretical = m.get("sm__maximum_warps_per_active_cycle_pct")
    if has_room and theoretical is not None and theoretical < 50 and (waves or 0) >= 1:
        limits = {
            key: m[f"launch__occupancy_limit_{key}"]
            for key in ("registers", "shared_mem", "warps", "blocks")
            if f"launch__occupancy_limit_{key}" in m
        }
        limiter = min(limits, key=limits.get) if limits else "unknown"
        regs = m.get("launch__registers_per_thread")
        add("occupancy-limited", f"theoretical occupancy {theoretical:.0f}%, limited by {limiter}"
            + (f" ({regs:.0f} registers/thread)" if regs is not None else "") + room)

    stalls = {name: m.get(f"smsp__average_warp_latency_issue_stalled_{name}.ratio", 0.0) for name in STALLS}
    total = sum(stalls.values())
    if has_room and total > 0:
        barrier = stalls["barrier"] / total
        if barrier > 0.2:
            add("barrier-stall", f"{barrier:.0%} of stall cycles at barriers{room}")
        memory = stalls["long_scoreboard"] / total
        l2 = m.get("lts__throughput.avg.pct_of_peak_sustained_elapsed")
        if memory > 0.5 and l2 is not None and l2 < 40:
            add("memory-latency-bound", f"{memory:.0%} of stall cycles waiting on global memory, L2 throughput {l2:.0f}%{room}")

    if headroom is not None and headroom < NEAR_FLOOR:
        add("near-floor", f"headroom {headroom:.2f}x over the measured floor")
    return findings


def summarize(findings: list[Finding]) -> tuple[str, ...]:
    """One line per symptom code, listing each shape's evidence, then the levers and the rule's basis."""
    by_code: dict[str, list[Finding]] = {}
    for finding in findings:
        by_code.setdefault(finding.code, []).append(finding)
    lines = []
    for code in RULES:
        group = by_code.get(code)
        if not group:
            continue
        rule = RULES[code]
        evidence = "; ".join(f"{f.shape}: {f.evidence}" for f in group)
        levers = " | ".join(f"{lever_id} ({text})" for lever_id, text in rule.levers)
        lines.append(f"[{code}] {evidence}. Try: {levers}. Basis: {rule.basis}")
    return tuple(lines)


def symptom_codes(symptoms) -> set[str]:
    """Rule codes named by `[code] ...` symptom lines; other symptom text is ignored."""
    codes = set()
    for line in symptoms:
        if line.startswith("[") and "]" in line:
            code = line[1:line.index("]")]
            if code in RULES:
                codes.add(code)
    return codes
