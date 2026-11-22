"""Fetch files from a HuggingFace dataset repository and verify them against
its published checksums."""

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm


STATE_DIR = ".download_state"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f, tqdm(total=path.stat().st_size, unit="B", unit_scale=True,
                                     desc=f"Verifying {path.name}") as bar:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
            bar.update(len(block))
    return h.hexdigest()


def download(repo_id: str, repo_path: str, work_dir: Path, revision=None) -> Path:
    """Fetch one file into work_dir, bypassing the HuggingFace cache so
    multi-gigabyte archives are not stored twice."""
    return Path(hf_hub_download(repo_id, repo_path, repo_type="dataset",
                                revision=revision, local_dir=work_dir))


def load_checksums(repo_id: str, work_dir: Path, revision=None) -> dict:
    """{repo path: sha256} for every published file of the repository."""
    sums = {}
    for line in download(repo_id, "checksums.sha256", work_dir, revision).read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        sums[path] = digest
    return sums


def published_digest(sums: dict, repo_id: str, repo_path: str) -> str:
    """The published checksum of one file, or an error naming the file when
    the repository does not publish it."""
    if repo_path not in sums:
        raise SystemExit(f"{repo_id} publishes no checksum for {repo_path}; check the "
                         f"repository revision or update MEMO")
    return sums[repo_path]


def fetch_verified(repo_id: str, repo_path: str, expected_sha: str, work_dir: Path,
                   revision=None) -> Path:
    path = download(repo_id, repo_path, work_dir, revision)
    actual = sha256_file(path)
    if actual != expected_sha:
        path.unlink()
        raise SystemExit(f"{repo_id}/{repo_path}: checksum mismatch ({actual} != "
                         f"{expected_sha}); rerun to fetch it again")
    return path


def work_dir_for(root: Path):
    """A temporary directory on the destination's filesystem, so verified
    files move into place without a cross-device copy."""
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root, prefix=".download_")


def record_path(root: Path, repo_id: str, repo_path: str) -> Path:
    return root / STATE_DIR / repo_id.replace("/", "__") / f"{repo_path}.json"


def already_unpacked(root: Path, repo_id: str, repo_path: str, digest: str) -> bool:
    """True when this archive version was unpacked under root and every file
    it holds is still present at its recorded size."""
    record = record_path(root, repo_id, repo_path)
    if not record.is_file():
        return False
    state = json.loads(record.read_text())
    return state["sha256"] == digest and all(
        (root / name).is_file() and (root / name).stat().st_size == size
        for name, size in state["sizes"].items())


def unpack(archive: Path, root: Path, rename) -> dict:
    """Extract every member to root / rename(member name), letting zip check
    each file's CRC as it is read, and return {destination: size}."""
    sizes = {}
    with zipfile.ZipFile(archive) as zf:
        for info in tqdm(zf.infolist(), desc=f"Unpacking {archive.name}"):
            if info.is_dir():
                continue
            dest = rename(info.filename)
            target = root / dest
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                while block := src.read(1 << 24):
                    out.write(block)
            sizes[dest] = info.file_size
    return sizes


def write_record(root: Path, repo_id: str, repo_path: str, digest: str, sizes: dict):
    record = record_path(root, repo_id, repo_path)
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"sha256": digest, "sizes": sizes}))
