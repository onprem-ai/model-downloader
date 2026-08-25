import hashlib
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from opai_models.client import LicenseClient, ModelAccess, ModelDownloadError
from opai_models.download import DownloadCancelled, pull_file_with_state


def access(content: bytes, *, source: str = "source", checksum: str | None = None) -> ModelAccess:
    return ModelAccess(
        "file",
        "https://s3.example/file",
        len(content),
        source,
        "later",
        {"sha256": checksum or hashlib.sha256(content).hexdigest()},
        {},
    )


@pytest.mark.asyncio
async def test_external_state_download_records_only_durable_chunks(tmp_path: Path) -> None:
    content = b"abcdef"
    client = MagicMock(spec=LicenseClient)
    client.access = AsyncMock(return_value=access(content))
    completed = {0}
    events = []
    partial = tmp_path / "file"
    partial.write_bytes(b"abc" + b"\0" * 3)

    async def ranged(provider, descriptor, start, end, retries, timeout, should_cancel, **kwargs):
        os.pwrite(descriptor, content[start : end + 1], start)
        return end - start + 1

    with patch("opai_models.download._download_range", side_effect=ranged) as transfer:
        result = await pull_file_with_state(
            client,
            "example",
            "file",
            partial,
            expected_source_id="source",
            expected_size=6,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            completed_chunks=completed,
            mark_complete=lambda index, size, rate: events.append((index, size, rate)),
            chunk_size=3,
            workers=1,
        )
    assert result.read_bytes() == content
    assert transfer.call_count == 1
    assert len(events) == 1
    assert events[0][:2] == (1, 6)
    assert events[0][2] >= 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source_id": "different"}, "source model changed"),
        ({"size": 7}, "source model changed"),
        ({"checksums": {"sha256": "b" * 64}}, "checksum changed"),
    ],
)
@pytest.mark.asyncio
async def test_external_state_rejects_changed_source(tmp_path: Path, change, message: str) -> None:
    content = b"abcdef"
    current = access(content)
    client = MagicMock(spec=LicenseClient)
    client.access = AsyncMock(return_value=ModelAccess(**{**current.__dict__, **change}))
    with pytest.raises(ModelDownloadError, match=message):
        await pull_file_with_state(
            client,
            "example",
            current.path,
            tmp_path / "file",
            expected_source_id=current.source_id,
            expected_size=current.size,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            completed_chunks=set(),
            mark_complete=lambda *_: None,
            chunk_size=3,
        )


@pytest.mark.asyncio
async def test_external_state_rejects_invalid_chunks_and_final_hash(tmp_path: Path) -> None:
    content = b"abcdef"
    client = MagicMock(spec=LicenseClient)
    client.access = AsyncMock(return_value=access(content))
    arguments = dict(
        expected_source_id="source",
        expected_size=6,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        mark_complete=lambda *_: None,
        chunk_size=3,
    )
    with pytest.raises(ModelDownloadError, match="chunk state"):
        await pull_file_with_state(
            client, "example", "file", tmp_path / "file", completed_chunks={2}, **arguments
        )

    async def corrupt(provider, descriptor, start, end, retries, timeout, should_cancel, **kwargs):
        os.pwrite(descriptor, b"x" * (end - start + 1), start)
        return end - start + 1

    with (
        patch("opai_models.download._download_range", side_effect=corrupt),
        pytest.raises(ModelDownloadError, match="SHA-256"),
    ):
        await pull_file_with_state(
            client, "example", "file", tmp_path / "file", completed_chunks=set(), **arguments
        )


@pytest.mark.asyncio
async def test_external_state_accepts_missing_provider_checksum(tmp_path: Path) -> None:
    content = b"abcdef"
    client = MagicMock(spec=LicenseClient)
    current = access(content)
    client.access = AsyncMock(return_value=ModelAccess(**{**current.__dict__, "checksums": {}}))

    async def ranged(provider, descriptor, start, end, retries, timeout, should_cancel, **kwargs):
        os.pwrite(descriptor, content[start : end + 1], start)
        return end - start + 1

    with patch("opai_models.download._download_range", side_effect=ranged):
        result = await pull_file_with_state(
            client,
            "example",
            "file",
            tmp_path / "file",
            expected_source_id="source",
            expected_size=6,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            completed_chunks=set(),
            mark_complete=lambda *_: None,
            chunk_size=3,
        )
    assert result.read_bytes() == content


@pytest.mark.asyncio
async def test_external_state_propagates_cancellation(tmp_path: Path) -> None:
    content = b"abcdef"
    client = MagicMock(spec=LicenseClient)
    client.access = AsyncMock(return_value=access(content))
    with pytest.raises(DownloadCancelled):
        await pull_file_with_state(
            client,
            "example",
            "file",
            tmp_path / "file",
            expected_source_id="source",
            expected_size=6,
            expected_sha256=hashlib.sha256(content).hexdigest(),
            completed_chunks=set(),
            mark_complete=lambda *_: None,
            chunk_size=3,
            should_cancel=lambda: True,
        )
