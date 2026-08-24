import errno
import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opai_models.async_client import AsyncModelClient
from opai_models.client import ModelDownloadError
from opai_models.metadata import SourceDocument
from opai_models.signatures import SigstoreIdentity
from opai_models.snapshot import ModelFile, ModelSnapshot
from opai_models.sync import _files, _link_or_copy, _replace_directory, sync_model


def source() -> SourceDocument:
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


def snapshot(contents: dict[str, bytes]) -> ModelSnapshot:
    return ModelSnapshot.create(
        "example",
        [
            ModelFile(path, path, len(data), "source-" + path, hashlib.sha256(data).hexdigest())
            for path, data in contents.items()
        ],
        source(),
    )


def write_model(directory: Path, contents: dict[str, bytes], manifest: bytes | None) -> None:
    directory.mkdir()
    for relative, data in contents.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    if manifest is not None:
        (directory / "SHA256SUMS").write_bytes(manifest)


def client_for(snap: ModelSnapshot) -> AsyncModelClient:
    client = AsyncModelClient("https://license.example", lambda: "license", verify_signatures=False)
    client.snapshot_model = AsyncMock(return_value=snap)
    return client


@pytest.mark.asyncio
async def test_matching_manifest_reuses_without_hashing(tmp_path: Path) -> None:
    contents = {".source.json": b"source", "weights/a": b"payload"}
    snap = snapshot(contents)
    destination = tmp_path / "example"
    write_model(destination, contents, snap.sha256sums.encode())
    client = client_for(snap)
    with patch("opai_models.sync._file_sha256") as file_hash:
        result = await sync_model(client, "example", destination)
    file_hash.assert_not_called()
    assert result.reused_files == 2 and result.downloaded_files == 0
    assert destination.joinpath("weights/a").read_bytes() == b"payload"


@pytest.mark.asyncio
async def test_rehash_hashes_every_file_and_reuses_valid_files(tmp_path: Path) -> None:
    contents = {".source.json": b"source", "file": b"payload"}
    snap = snapshot(contents)
    destination = tmp_path / "example"
    write_model(destination, contents, snap.sha256sums.encode())
    client = client_for(snap)
    with patch(
        "opai_models.sync._file_sha256",
        side_effect=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    ) as file_hash:
        result = await sync_model(client, "example", destination, rehash=True)
    assert file_hash.call_count == 2
    assert result.rehashed_files == 2 and result.reused_files == 2


@pytest.mark.asyncio
async def test_changed_manifest_selectively_reuses_and_repairs(tmp_path: Path) -> None:
    old = {".source.json": b"source", "same": b"same", "changed": b"old"}
    current = {**old, "changed": b"new"}
    old_snapshot = snapshot(old)
    new_snapshot = snapshot(current)
    destination = tmp_path / "example"
    write_model(destination, old, old_snapshot.sha256sums.encode())
    (destination / "extra").write_bytes(b"keep")
    client = client_for(new_snapshot)

    def transfer(client, model_id, path, target, **kwargs):
        target.write_bytes(current[path])
        return target

    with patch("opai_models.sync.pull_file_with_state", side_effect=transfer) as download:
        result = await sync_model(client, "example", destination)
    assert download.call_count == 1
    assert download.call_args.args[2] == "changed"
    assert result.reused_files == 2 and result.downloaded_files == 1
    assert (destination / "changed").read_bytes() == b"new"
    assert (destination / "extra").read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_missing_manifest_hashes_existing_files_and_delete_removes_extras(
    tmp_path: Path,
) -> None:
    contents = {".source.json": b"source", "file": b"payload"}
    snap = snapshot(contents)
    destination = tmp_path / "example"
    write_model(destination, contents, None)
    (destination / "extra").write_bytes(b"remove")
    client = client_for(snap)
    result = await sync_model(client, "example", destination, delete=True)
    assert result.rehashed_files == 2
    assert result.reused_files == 2
    assert result.deleted_files == 1
    assert not (destination / "extra").exists()
    assert (destination / "SHA256SUMS").read_bytes() == snap.sha256sums.encode()


@pytest.mark.asyncio
async def test_corrupt_file_is_downloaded_and_old_directory_survives_failure(
    tmp_path: Path,
) -> None:
    contents = {".source.json": b"source", "file": b"correct"}
    snap = snapshot(contents)
    destination = tmp_path / "example"
    write_model(destination, {**contents, "file": b"corrupt"}, None)
    client = client_for(snap)
    with patch("opai_models.sync.pull_file_with_state", side_effect=RuntimeError("network")):
        with pytest.raises(RuntimeError, match="network"):
            await sync_model(client, "example", destination)
    assert (destination / "file").read_bytes() == b"corrupt"
    assert destination.with_name(".example.sync.partial").is_dir()


def test_replace_directory_restores_original_on_publish_failure(tmp_path: Path) -> None:
    destination = tmp_path / "example"
    staging = tmp_path / ".example.sync.partial"
    backup = tmp_path / ".example.sync.previous"
    destination.mkdir()
    staging.mkdir()
    (destination / "old").write_text("old")
    original_replace = __import__("os").replace
    calls = 0

    def replace(source: Path, target: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("publish failed")
        original_replace(source, target)

    with patch("opai_models.sync.os.replace", side_effect=replace):
        with pytest.raises(OSError, match="publish failed"):
            _replace_directory(staging, destination, backup)
    assert (destination / "old").read_text() == "old"


@pytest.mark.asyncio
async def test_sync_reuses_verified_staging_after_interruption(tmp_path: Path) -> None:
    contents = {".source.json": b"source", "file": b"payload"}
    snap = snapshot(contents)
    destination = tmp_path / "example"
    write_model(destination, {".source.json": b"source"}, None)
    staging = tmp_path / ".example.sync.partial"
    write_model(staging, {"file": b"payload"}, None)
    client = client_for(snap)
    with patch("opai_models.sync.pull_file_with_state") as download:
        result = await sync_model(client, "example", destination)
    download.assert_not_called()
    assert result.reused_files == 2
    assert (destination / "file").read_bytes() == b"payload"


@pytest.mark.asyncio
async def test_sync_recovers_backup_and_rejects_unsafe_staging(tmp_path: Path) -> None:
    contents = {".source.json": b"source", "file": b"payload"}
    client = client_for(snapshot(contents))
    destination = tmp_path / "example"
    backup = tmp_path / ".example.sync.previous"
    write_model(backup, contents, None)
    result = await sync_model(client, "example", destination)
    assert result.destination == destination
    assert not backup.exists()

    staging = tmp_path / ".example.sync.partial"
    destination.mkdir(exist_ok=True)
    staging.write_text("unsafe")
    with pytest.raises(ModelDownloadError, match="staging"):
        await sync_model(client, "example", destination)


def test_files_rejects_symlinks_and_copy_fallback(tmp_path: Path) -> None:
    source_file = tmp_path / "source"
    source_file.write_bytes(b"data")
    link = tmp_path / "link"
    link.symlink_to(source_file)
    with pytest.raises(ModelDownloadError, match="symbolic"):
        _files(tmp_path)
    destination = tmp_path / "nested" / "copy"
    with patch("opai_models.sync.os.link", side_effect=OSError(errno.EXDEV, "cross-device")):
        _link_or_copy(source_file, destination)
    assert destination.read_bytes() == b"data"


@pytest.mark.asyncio
async def test_sync_progress_and_signature_verification(tmp_path: Path) -> None:
    contents = {".source.json": b"source", "file": b"payload"}
    snap = snapshot(contents)
    destination = tmp_path / "example"
    write_model(destination, contents, snap.sha256sums.encode())
    (destination / "SHA256SUMS.sigstore.json").write_bytes(b"old-bundle")
    client = AsyncModelClient(
        "https://license.example",
        lambda: "license",
        sigstore_identity="trusted@example.com",
        sigstore_issuer="https://issuer.example",
    )
    client.snapshot_model = AsyncMock(return_value=snap)
    raw = MagicMock()
    raw.read_small.return_value = b"new-bundle"
    client._client = MagicMock(return_value=raw)
    events = []
    with patch("opai_models.sync.verify_sigstore_bundle") as verify:
        result = await sync_model(client, "example", destination, progress=events.append)
    assert result.reused_files == 2
    assert [event["event"] for event in events] == ["file_reused", "file_reused"]
    verify.assert_called_once_with(
        snap.sha256sums.encode(),
        b"new-bundle",
        SigstoreIdentity("trusted@example.com", "https://issuer.example"),
        offline=False,
    )
    assert (destination / "SHA256SUMS.sigstore.json").read_bytes() == b"new-bundle"


@pytest.mark.asyncio
async def test_async_client_sync_wrapper_forwards_options(tmp_path: Path) -> None:
    client = client_for(snapshot({".source.json": b"source", "file": b"payload"}))
    expected = MagicMock()
    with patch("opai_models.sync.sync_model", new=AsyncMock(return_value=expected)) as operation:
        assert (
            await client.sync_model(
                "example",
                tmp_path / "example",
                rehash=True,
                delete=True,
                workers=3,
            )
            is expected
        )
    assert operation.call_args.kwargs["rehash"] is True
    assert operation.call_args.kwargs["delete"] is True
    assert operation.call_args.kwargs["workers"] == 3


@pytest.mark.asyncio
async def test_sync_rejects_missing_remote_manifest(tmp_path: Path) -> None:
    snap = ModelSnapshot.create(
        "example",
        [ModelFile("file", "file", 1, "source", None)],
        source(),
    )
    destination = tmp_path / "example"
    destination.mkdir()
    client = client_for(snap)
    with pytest.raises(ModelDownloadError, match="remote SHA256SUMS"):
        await sync_model(client, "example", destination)


@pytest.mark.asyncio
async def test_sync_rejects_invalid_destination_and_backup(tmp_path: Path) -> None:
    client = client_for(snapshot({".source.json": b"source", "file": b"payload"}))
    with pytest.raises(ModelDownloadError, match="existing"):
        await sync_model(client, "example", tmp_path / "missing")
    destination = tmp_path / "example"
    destination.mkdir()
    destination.with_name(".example.sync.previous").mkdir()
    with pytest.raises(ModelDownloadError, match="backup"):
        await sync_model(client, "example", destination)
