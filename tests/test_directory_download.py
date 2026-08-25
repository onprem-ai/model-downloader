import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from conftest import license_key

from opai_models.async_client import AsyncModelClient, _completed_directory_matches
from opai_models.manager import SQLiteQueueStore
from opai_models.metadata import SourceDocument
from opai_models.signatures import SigstoreIdentity
from opai_models.snapshot import ModelFile, ModelSnapshot


def snapshot(contents: dict[str, bytes]) -> ModelSnapshot:
    source_data = json.dumps(provenance().to_dict(), ensure_ascii=False, indent=2).encode() + b"\n"
    complete = {".source.json": source_data, **contents}
    contents.clear()
    contents.update(complete)
    return ModelSnapshot.create(
        "example",
        [
            ModelFile(
                path,
                path,
                len(data),
                "source-" + path,
                hashlib.sha256(data).hexdigest(),
            )
            for path, data in contents.items()
        ],
        provenance(),
    )


def provenance() -> SourceDocument:
    return SourceDocument.from_dict(
        {
            "schema_version": 1,
            "source": {
                "provider": "huggingface",
                "repository": "owner/model",
                "revision": "a" * 40,
            },
        }
    )


def test_completed_directory_signature_and_checksum_controls(tmp_path: Path) -> None:
    contents = {"file": b"correct"}
    snap = snapshot(contents)
    destination = tmp_path / "model"
    destination.mkdir()
    for path, data in contents.items():
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (destination / "SHA256SUMS").write_text(snap.sha256sums)
    (destination / "SHA256SUMS.sigstore.json").write_bytes(b"bundle")
    job = MagicMock(snapshot_sha256=snap.snapshot_sha256)
    files = [
        MagicMock(relative_path=item.relative_path, expected_sha256=item.sha256)
        for item in snap.files
    ]
    identity = SigstoreIdentity("identity", "https://issuer")
    with __import__("unittest.mock").mock.patch(
        "opai_models.async_client.verify_sigstore_bundle"
    ) as verify:
        assert _completed_directory_matches(
            destination,
            job,
            files,
            snap.source,
            verify_checksums=True,
            verify_signatures=True,
            trusted_identity=identity,
            sigstore_offline=True,
        )
    verify.assert_called_once_with(snap.sha256sums.encode(), b"bundle", identity, offline=True)

    (destination / "file").write_bytes(b"corrupt")
    assert (
        _completed_directory_matches(
            destination,
            job,
            files,
            snap.source,
            verify_checksums=False,
            verify_signatures=False,
            trusted_identity=None,
            sigstore_offline=False,
        )
        is False
    )  # The unrequested signature is an unexpected local file.
    (destination / "SHA256SUMS.sigstore.json").unlink()
    assert _completed_directory_matches(
        destination,
        job,
        files,
        snap.source,
        verify_checksums=False,
        verify_signatures=False,
        trusted_identity=None,
        sigstore_offline=False,
    )


@pytest.mark.asyncio
async def test_pull_model_stages_verifies_and_atomically_publishes(tmp_path: Path) -> None:
    contents = {"config.json": b"config", "weights/model.bin": b"weights"}
    snap = snapshot(contents)
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed
    token = claimed[1]
    assert store.save_snapshot(job.id, token, snap)

    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)

    async def pull_file(client, model_id, path, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents[path])
        await kwargs["mark_complete"](0, destination.stat().st_size, 123)
        return destination

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("opai_models.async_client.pull_file_with_state", pull_file)
        result = await client._pull_job(job.id, tmp_path / "final", store, token, workers=2)
    assert result == (tmp_path / "final").resolve()
    assert (result / "config.json").read_bytes() == b"config"
    assert (result / "weights/model.bin").read_bytes() == b"weights"
    assert (
        json.loads((result / ".source.json").read_text())["source"]["repository"] == "owner/model"
    )
    assert (result / "SHA256SUMS").read_text() == snap.sha256sums
    assert all(item.state == "completed" for item in store.files(job.id))
    assert store.get(job.id).state == "verifying"
    assert not result.with_name(f".final.{job.id}.partial").exists()


@pytest.mark.asyncio
async def test_pull_model_without_verification_uses_inventory_without_manifest(
    tmp_path: Path,
) -> None:
    source_data = json.dumps(provenance().to_dict(), ensure_ascii=False, indent=2).encode() + b"\n"
    contents = {".source.json": source_data, "file": b"payload"}
    snap = ModelSnapshot.create(
        "example",
        [
            ModelFile(path, path, len(data), "source-" + path, None)
            for path, data in contents.items()
        ],
        provenance(),
    )
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed and store.save_snapshot(job.id, claimed[1], snap)
    client = AsyncModelClient(
        "https://license.example",
        license_key,
        verify_checksums=False,
        verify_signatures=False,
    )

    async def pull_file(client, model_id, path, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents[path])
        await kwargs["mark_complete"](0, destination.stat().st_size, 123)
        assert kwargs["expected_sha256"] is None
        assert kwargs["verify_checksum"] is False
        return destination

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("opai_models.async_client.pull_file_with_state", pull_file)
        result = await client._pull_job(job.id, tmp_path / "final", store, claimed[1])
    assert result == (tmp_path / "final").resolve()
    assert (result / "file").read_bytes() == b"payload"
    assert not (result / "SHA256SUMS").exists()
    assert all(item.expected_sha256 is None for item in store.files(job.id))


@pytest.mark.asyncio
async def test_pull_model_rejects_checksum_and_preserves_staging(tmp_path: Path) -> None:
    contents = {"file": b"correct"}
    snap = snapshot(contents)
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed and store.save_snapshot(job.id, claimed[1], snap)
    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)

    async def corrupt(client, model_id, path, destination, **kwargs):
        relative = path
        destination.write_bytes(b"wrong" if relative == "file" else contents[relative])
        return destination

    file = next(item for item in store.files(job.id) if item.relative_path == "file")
    assert store.prepare_chunks(file.id, claimed[1], file.expected_size, file.expected_size)
    assert store.record_progress(
        file.id,
        claimed[1],
        {"event": "chunk_complete", "chunk": 0, "completed_bytes": file.expected_size},
    )
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        pytest.raises(Exception, match="SHA-256"),
    ):
        monkeypatch.setattr("opai_models.async_client.pull_file_with_state", corrupt)
        await client._pull_job(job.id, tmp_path / "final", store, claimed[1])
    assert not (tmp_path / "final").exists()
    assert (tmp_path / f".final.{job.id}.partial/file").exists()
    assert store.completed_chunks(file.id) == set()


@pytest.mark.asyncio
async def test_pull_model_rejects_unrelated_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "final"
    destination.mkdir()
    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)
    store = MagicMock()
    store.get.return_value = MagicMock(snapshot_sha256="sha256:" + "a" * 64)
    store.files.return_value = []
    with pytest.raises(Exception, match="does not match"):
        await client._pull_job("job", destination, store, "token")


@pytest.mark.asyncio
async def test_pull_model_rejects_symlink_staging_path(tmp_path: Path) -> None:
    snap = snapshot({"file": b"correct"})
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed and store.save_snapshot(job.id, claimed[1], snap)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / f".final.{job.id}.partial").symlink_to(outside, target_is_directory=True)
    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)
    with pytest.raises(Exception, match="symbolic link"):
        await client._pull_job(job.id, tmp_path / "final", store, claimed[1])


@pytest.mark.asyncio
async def test_pull_model_rejects_staging_file_symlink(tmp_path: Path) -> None:
    snap = snapshot({"file": b"correct"})
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed and store.save_snapshot(job.id, claimed[1], snap)
    staging = tmp_path / f".final.{job.id}.partial"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    (staging / ".source.json").symlink_to(outside)
    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)
    with pytest.raises(Exception, match="staging file"):
        await client._pull_job(job.id, tmp_path / "final", store, claimed[1])
    assert outside.read_text() == "untouched"


@pytest.mark.asyncio
async def test_pull_model_rejects_nested_staging_symlink(tmp_path: Path) -> None:
    snap = snapshot({"nested/file": b"correct"})
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed and store.save_snapshot(job.id, claimed[1], snap)
    staging = tmp_path / f".final.{job.id}.partial"
    staging.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (staging / "nested").symlink_to(outside, target_is_directory=True)
    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)

    async def pull_source(client, model_id, path, destination, **kwargs):
        destination.write_bytes(contents[path])
        return destination

    contents = {item.relative_path: b"" for item in snap.files}
    contents[".source.json"] = (
        json.dumps(provenance().to_dict(), ensure_ascii=False, indent=2).encode() + b"\n"
    )
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        pytest.raises(Exception, match="symbolic links"),
    ):
        monkeypatch.setattr("opai_models.async_client.pull_file_with_state", pull_source)
        await client._pull_job(job.id, tmp_path / "final", store, claimed[1])


@pytest.mark.asyncio
async def test_pull_model_rejects_lost_lease_before_publish(tmp_path: Path) -> None:
    contents = {"file": b"correct"}
    snap = snapshot(contents)
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed and store.save_snapshot(job.id, claimed[1], snap)

    async def pull_file(client, model_id, path, destination, **kwargs):
        data = contents[path]
        destination.write_bytes(data)
        await kwargs["mark_complete"](0, len(data), 1)
        return destination

    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)
    original = store.cancellation_requested
    checks = iter([True])
    store.cancellation_requested = lambda *args: next(checks, original(*args))
    with (
        pytest.MonkeyPatch.context() as monkeypatch,
        pytest.raises(Exception, match="lease was lost"),
    ):
        monkeypatch.setattr("opai_models.async_client.pull_file_with_state", pull_file)
        await client._pull_job(job.id, tmp_path / "final", store, claimed[1])
    assert not (tmp_path / "final").exists()


@pytest.mark.asyncio
async def test_pull_model_recovers_after_rename_before_queue_finish(tmp_path: Path) -> None:
    contents = {"file": b"correct"}
    snap = snapshot(contents)
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue(snap.model_id, str(tmp_path / "final"))
    claimed = store.claim("worker", 60)
    assert claimed and store.save_snapshot(job.id, claimed[1], snap)
    destination = tmp_path / "final"
    destination.mkdir()
    (destination / "file").write_bytes(b"correct")
    (destination / "SHA256SUMS").write_text(snap.sha256sums)
    (destination / ".source.json").write_bytes(contents[".source.json"])
    client = AsyncModelClient("https://license.example", license_key, verify_signatures=False)
    assert await client._pull_job(job.id, destination, store, claimed[1]) == destination
    assert all(item.state == "completed" for item in store.files(job.id))

    (destination / "unexpected").write_text("extra")
    with pytest.raises(Exception, match="does not match"):
        await client._pull_job(job.id, destination, store, claimed[1])
