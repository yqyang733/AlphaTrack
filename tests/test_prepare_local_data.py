import hashlib
import json

import pytest

from scripts.prepare_local_data import check, load_manifest, restore


def snapshot(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    records = []
    for name, content in [("data/sample.csv", b"date,nav\n2020-01-01,1\n"),
                          ("vendor/qlab/data_files/dividend/fund/sample.csv", b"dividend\n0.1\n")]:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        records.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": records}))
    return source, destination, manifest, load_manifest(manifest)


def test_restore_missing_snapshot_and_verify_without_overwriting(tmp_path):
    source, destination, _, records = snapshot(tmp_path)
    assert restore(source, destination, records) == 2
    assert check(destination, records) == 2
    assert restore(source, destination, records) == 0
    for record in records:
        assert (destination / record["path"]).read_bytes() == (source / record["path"]).read_bytes()


def test_source_hash_failure_prevents_all_copying(tmp_path):
    source, destination, _, records = snapshot(tmp_path)
    (source / records[-1]["path"]).write_bytes(b"unexpected data")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        restore(source, destination, records)
    assert not destination.exists()


def test_conflicting_destination_is_never_overwritten(tmp_path):
    source, destination, _, records = snapshot(tmp_path)
    existing = destination / records[-1]["path"]
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"keep my existing data")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        restore(source, destination, records)
    assert existing.read_bytes() == b"keep my existing data"
    assert not (destination / records[0]["path"]).exists()


@pytest.mark.parametrize("name", ["data/../../outside", "/data/file", "../data/file",
                                 "data/./file", "data//file", "scripts/file", "data\\file"])
def test_manifest_rejects_paths_outside_snapshot_directories(tmp_path, name):
    _, _, manifest, records = snapshot(tmp_path)
    records[0]["path"] = name
    manifest.write_text(json.dumps({"schema_version": 1, "files": records}))
    with pytest.raises(ValueError, match="Unsafe snapshot path"):
        load_manifest(manifest)


def test_destination_symlink_cannot_redirect_restore(tmp_path):
    source, destination, _, records = snapshot(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    destination.mkdir()
    (destination / "data").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        restore(source, destination, records)
    assert list(outside.iterdir()) == []
