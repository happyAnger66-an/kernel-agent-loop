"""模块: kernel_agent_loop.evald.bundle
功能: 候选包只带实现目录，越权成员整包拒收，实现目录整体替换。
"""

import io
import json
import shutil
import tarfile
from pathlib import Path

import pytest

from kernel_agent_loop.evald.bundle import BundleError, make_bundle, overlay_bundle

META = {"campaign": "rms-1", "commit": "c1", "parent": "c0", "op": "rmsnorm", "backend": "cuda",
        "lever": "vectorize-loads", "hypothesis": "use 128-bit loads"}
IMPL = Path("ops/rmsnorm/impls/cuda")


@pytest.fixture
def trees(op_repo: Path, tmp_path: Path):
    trusted = tmp_path / "trusted"
    shutil.copytree(op_repo, trusted, ignore=shutil.ignore_patterns(".git"))
    dev = op_repo
    (dev / IMPL / "kernel.cu").write_text("// v1\n")
    (dev / "ops/rmsnorm/spec.py").write_text("TAMPERED\n")
    return trusted, dev, tmp_path / "work"


def raw_bundle(members: dict[str, bytes], meta=META, symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in {"meta.json": json.dumps(meta).encode(), **members}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if symlink:
            info = tarfile.TarInfo(symlink)
            info.type = tarfile.SYMTYPE
            info.linkname = "/etc/passwd"
            tar.addfile(info)
    return buf.getvalue()


# 用例: 候选包带上实现改动，带不走对 spec.py 的篡改
# 参数: trees 是可信树、开发机工作区和评测工作目录
def test_bundle_carries_impl_changes_but_not_spec_changes(trees):
    trusted, dev, work = trees
    meta = overlay_bundle(make_bundle(dev, "rmsnorm", "cuda", META), trusted, work, op="rmsnorm", backend="cuda")
    assert meta == META
    assert (work / IMPL / "kernel.cu").read_text() == "// v1\n"
    assert (work / "ops/rmsnorm/spec.py").read_text() == "TRUSTED_SPEC\n"


# 用例: 展开候选包不改动可信源码树
# 参数: trees 是可信树、开发机工作区和评测工作目录
def test_overlay_leaves_trusted_tree_untouched(trees):
    trusted, dev, work = trees
    overlay_bundle(make_bundle(dev, "rmsnorm", "cuda", META), trusted, work, op="rmsnorm", backend="cuda")
    assert (trusted / IMPL / "kernel.cu").read_text() == "// v0\n"


# 用例: 实现目录整体替换，候选删掉的文件不会残留
# 参数: trees 是可信树、开发机工作区和评测工作目录
def test_impl_dir_is_replaced_not_merged(trees):
    trusted, dev, work = trees
    (dev / IMPL / "launch.py").unlink()
    overlay_bundle(make_bundle(dev, "rmsnorm", "cuda", META), trusted, work, op="rmsnorm", backend="cuda")
    assert not (work / IMPL / "launch.py").exists()


# 用例: 包里出现实现目录以外的成员时，整个包拒收且工作目录保持不存在
# 参数: trees 是可信树、开发机工作区和评测工作目录；name 是越权成员路径，覆盖契约文件、路径穿越、绝对路径和测试文件
@pytest.mark.parametrize("name", ["ops/rmsnorm/spec.py", "impl/../../spec.py", "/tmp/evil", "tests/x.py"])
def test_members_outside_impl_reject_the_whole_bundle(trees, name):
    trusted, _, work = trees
    bundle = raw_bundle({"impl/kernel.cu": b"// v1\n", name: b"evil"})
    with pytest.raises(BundleError):
        overlay_bundle(bundle, trusted, work, op="rmsnorm", backend="cuda")
    assert not work.exists()


# 用例: 包里的符号链接成员导致整个包拒收
# 参数: trees 是可信树、开发机工作区和评测工作目录
def test_symlink_member_rejects_the_bundle(trees):
    trusted, _, work = trees
    bundle = raw_bundle({"impl/kernel.cu": b"// v1\n"}, symlink="impl/passwd")
    with pytest.raises(BundleError):
        overlay_bundle(bundle, trusted, work, op="rmsnorm", backend="cuda")


# 用例: meta.json 里的算子和 campaign 不一致时拒收
# 参数: trees 是可信树、开发机工作区和评测工作目录
def test_meta_pointing_at_another_op_is_rejected(trees):
    trusted, _, work = trees
    bundle = raw_bundle({"impl/kernel.cu": b"// v1\n"}, meta={**META, "op": "softmax"})
    with pytest.raises(BundleError):
        overlay_bundle(bundle, trusted, work, op="rmsnorm", backend="cuda")


# 用例: 候选包带上 incumbent 时展开到工作区 _incumbent，候选实现目录不受影响
# 参数: trees 是可信树、开发机工作区和评测工作目录
def test_bundle_carries_incumbent_separately(trees):
    trusted, dev, work = trees
    bundle = make_bundle(dev, "rmsnorm", "cuda", META, incumbent={"kernel.cu": b"// v0\n"})
    overlay_bundle(bundle, trusted, work, op="rmsnorm", backend="cuda")
    assert (work / "_incumbent/kernel.cu").read_text() == "// v0\n"
    assert (work / IMPL / "kernel.cu").read_text() == "// v1\n"


# 用例: incumbent 成员同样拒收越界路径
# 参数: trees 是可信树、开发机工作区和评测工作目录
def test_incumbent_member_with_parent_path_is_rejected(trees):
    trusted, _, work = trees
    with pytest.raises(BundleError):
        overlay_bundle(raw_bundle({"incumbent/../evil": b"x"}), trusted, work, op="rmsnorm", backend="cuda")
