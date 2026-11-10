"""Verify or restore the recorded local data snapshot without downloading data."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DATA_PREFIXES = ("data/", "vendor/qlab/data_files/dividend/")


def load_manifest(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("files"), list):
        raise ValueError("Unsupported data snapshot manifest")
    records = payload["files"]
    seen = set()
    for record in records:
        name = record.get("path", "")
        if (not isinstance(name, str) or not name or "\\" in name
                or PurePosixPath(name).is_absolute()
                or any(part in {"", ".", ".."} for part in name.split("/"))
                or not name.startswith(DATA_PREFIXES)):
            raise ValueError(f"Unsafe snapshot path: {name!r}")
        if name in seen:
            raise ValueError(f"Duplicate snapshot path: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", record.get("sha256", "")):
            raise ValueError(f"Invalid SHA256 for {name}")
        seen.add(name)
    if not records:
        raise ValueError("Data snapshot manifest is empty")
    return records


def safe_path(root, name):
    """Reject symlinks as well as paths outside the chosen repository."""
    root = Path(root).resolve()
    candidate = root
    for part in PurePosixPath(name).parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"Symlink is not allowed in snapshot path: {candidate}")
    if not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Snapshot path leaves repository: {name}")
    return candidate


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_file(path, expected):
    if not path.is_file():
        raise ValueError(f"Missing data file: {path}")
    if sha256(path) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")


def check(root, records):
    for record in records:
        verify_file(safe_path(root, record["path"]), record["sha256"])
    return len(records)


def restore(source_root, destination_root, records):
    # Finish the entire source and destination preflight before creating files.
    check(source_root, records)
    pending = []
    for record in records:
        destination = safe_path(destination_root, record["path"])
        if destination.exists():
            verify_file(destination, record["sha256"])
        else:
            pending.append(record)
    for record in pending:
        source = safe_path(source_root, record["path"])
        destination = safe_path(destination_root, record["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".snapshot-", delete=False) as stream:
                temporary = Path(stream.name)
                with source.open("rb") as original:
                    shutil.copyfileobj(original, stream)
            # Recheck the copied bytes before publishing, in case the source changed.
            verify_file(temporary, record["sha256"])
            safe_path(destination_root, record["path"])
            # An exclusive hard link publishes atomically and never overwrites a file.
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    check(destination_root, records)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify all local snapshot files")
    action.add_argument("--from", dest="source", type=Path, metavar="ALPHATRACK_DIR",
                        help="restore missing files from an existing AlphaTrack directory")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/data_snapshot.json")
    args = parser.parse_args()
    try:
        records = load_manifest(args.manifest)
        if args.check:
            print(f"Data snapshot verified: {check(ROOT, records)} files")
        else:
            count = restore(args.source, ROOT, records)
            print(f"Data snapshot verified: {len(records)} files; restored {count} missing files")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Data snapshot error: {exc}\n")


if __name__ == "__main__":
    main()
