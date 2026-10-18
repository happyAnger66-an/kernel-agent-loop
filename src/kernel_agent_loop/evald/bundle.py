import io
import json
import shutil
import tarfile
from pathlib import Path


class BundleError(Exception):
    pass


def make_bundle(repo: Path, op: str, backend: str, meta: dict) -> bytes:
    impl = repo / "ops" / op / "impls" / backend
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        payload = json.dumps(meta).encode()
        info = tarfile.TarInfo("meta.json")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
        for path in sorted(p for p in impl.rglob("*") if p.is_file()):
            tar.add(path, arcname=f"impl/{path.relative_to(impl).as_posix()}")
    return buf.getvalue()


def overlay_bundle(data: bytes, trusted: Path, work: Path, *, op: str, backend: str) -> dict:
    meta, files = _read_bundle(data, op=op, backend=backend)
    shutil.copytree(trusted, work)
    dest = work / "ops" / op / "impls" / backend
    shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for rel, content in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return meta


def _read_bundle(data: bytes, *, op: str, backend: str) -> tuple[dict, dict[str, bytes]]:
    files: dict[str, bytes] = {}
    meta = None
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if _rejected_member(member):
                raise BundleError(f"rejected bundle member {member.name}")
            if member.name == "meta.json":
                meta = json.loads(tar.extractfile(member).read())
            else:
                files[member.name.removeprefix("impl/")] = tar.extractfile(member).read()
    if meta is None or meta.get("op") != op or meta.get("backend") != backend:
        raise BundleError(f"bundle meta does not match {op}/{backend}")
    return meta, files


def _rejected_member(member: tarfile.TarInfo) -> bool:
    name = member.name
    if member.issym() or member.islnk():
        return True
    if name.startswith("/") or ".." in Path(name).parts:
        return True
    if name == "meta.json":
        return not member.isfile()
    if name.startswith("impl/") and name != "impl/" and member.isfile():
        return False
    return True
