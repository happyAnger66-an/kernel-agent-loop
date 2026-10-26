import io
import json
import shutil
import tarfile
from pathlib import Path


class BundleError(Exception):
    pass


INCUMBENT_DIR = "_incumbent"


def make_bundle(
    repo: Path, op: str, backend: str, meta: dict, *, incumbent: dict[str, bytes] | None = None
) -> bytes:
    """Pack the working-tree impl dir; `incumbent` maps impl-relative paths to the best version's files."""
    impl = repo / "ops" / op / "impls" / backend
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        _add_bytes(tar, "meta.json", json.dumps(meta).encode())
        for path in sorted(p for p in impl.rglob("*") if p.is_file()):
            tar.add(path, arcname=f"impl/{path.relative_to(impl).as_posix()}")
        for rel, content in sorted((incumbent or {}).items()):
            _add_bytes(tar, f"incumbent/{rel}", content)
    return buf.getvalue()


def overlay_bundle(data: bytes, trusted: Path, work: Path, *, op: str, backend: str) -> dict:
    """Copy trusted into work, replace the impl dir, and unpack any incumbent into work/_incumbent."""
    meta, files, incumbent = _read_bundle(data, op=op, backend=backend)
    shutil.copytree(trusted, work)
    dest = work / "ops" / op / "impls" / backend
    shutil.rmtree(dest)
    _write_tree(dest, files)
    if incumbent:
        _write_tree(work / INCUMBENT_DIR, incumbent)
    return meta


def _add_bytes(tar: tarfile.TarFile, name: str, content: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(content)
    tar.addfile(info, io.BytesIO(content))


def _write_tree(dest: Path, files: dict[str, bytes]) -> None:
    dest.mkdir(parents=True)
    for rel, content in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _read_bundle(data: bytes, *, op: str, backend: str) -> tuple[dict, dict[str, bytes], dict[str, bytes]]:
    files: dict[str, bytes] = {}
    incumbent: dict[str, bytes] = {}
    meta = None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if _rejected_member(member):
                raise BundleError(f"rejected bundle member {member.name}")
            if member.name == "meta.json":
                meta = json.loads(tar.extractfile(member).read())
            elif member.name.startswith("incumbent/"):
                incumbent[member.name.removeprefix("incumbent/")] = tar.extractfile(member).read()
            else:
                files[member.name.removeprefix("impl/")] = tar.extractfile(member).read()
    if meta is None or meta.get("op") != op or meta.get("backend") != backend:
        raise BundleError(f"bundle meta does not match {op}/{backend}")
    return meta, files, incumbent


def _rejected_member(member: tarfile.TarInfo) -> bool:
    name = member.name
    if member.issym() or member.islnk():
        return True
    if name.startswith("/") or ".." in Path(name).parts:
        return True
    if name == "meta.json":
        return not member.isfile()
    for prefix in ("impl/", "incumbent/"):
        if name.startswith(prefix) and name != prefix and member.isfile():
            return False
    return True
