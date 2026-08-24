import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from opai_models.download import DownloadCancelled
from opai_models.manager import DownloadManager, SQLiteQueueStore
from opai_models.metadata import SourceDocument
from opai_models.snapshot import ModelFile, ModelSnapshot


def snapshot() -> ModelSnapshot:
    source = SourceDocument.from_dict(
        {
            "schema_version": 1,
            "source": {
                "provider": "huggingface",
                "repository": "owner/model",
                "revision": "a" * 40,
            },
        }
    )
    return ModelSnapshot.create(
        "example",
        [ModelFile("a", "a", 1, "source", "a" * 64)],
        source,
    )


def manager(tmp_path: Path, client=None, **kwargs) -> DownloadManager:
    kwargs.setdefault("poll_interval", 0.01)
    return DownloadManager(
        tmp_path / "queue.sqlite",
        tmp_path / "models",
        client or MagicMock(),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_manager_public_api_and_context_lifecycle(tmp_path: Path) -> None:
    value = manager(tmp_path)
    job = await value.enqueue("example")
    assert (await value.get(job.id)).model_id == "example"
    assert (await value.list(state="queued"))[0].id == job.id
    assert (await value.cancel(job.id)).state == "cancelled"
    assert (await value.retry(job.id)).state == "queued"
    await value.cancel(job.id)
    async with value:
        await value.start()  # idempotent
    assert not value._tasks
    with pytest.raises(ValueError, match="inside"):
        await value.enqueue("example", "../escape")
    with pytest.raises(ValueError, match="max_attempts"):
        await value.enqueue("example", "other", max_attempts=0)


def test_manager_rejects_bad_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        manager(tmp_path, max_concurrent_downloads=0)
    with pytest.raises(ValueError):
        manager(tmp_path, lease_seconds=29)
    with pytest.raises(ValueError):
        manager(tmp_path, poll_interval=0)
    with pytest.raises(ValueError):
        manager(tmp_path, max_attempts=0)
    with pytest.raises(ValueError):
        manager(tmp_path, chunk_size=100)
    with pytest.raises(ValueError):
        manager(tmp_path, range_workers=33)


@pytest.mark.asyncio
async def test_worker_completes_job(tmp_path: Path) -> None:
    client = MagicMock()
    snap = snapshot()
    client.snapshot_model = AsyncMock(return_value=snap)

    async def complete(job_id, destination, store, token, **kwargs):
        file = store.files(job_id)[0]
        store.update_file(
            file.id,
            token,
            completed_bytes=file.expected_size,
            state="completed",
            sha256=file.expected_sha256,
        )
        store.mark_verifying(job_id, token)
        return destination

    client._pull_job = AsyncMock(side_effect=complete)
    value = manager(tmp_path, client=client, max_concurrent_downloads=1)
    job = await value.enqueue("example")
    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()
    assert result.state == "completed"
    client.snapshot_model.assert_awaited_once()
    client._pull_job.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [RuntimeError("secret-url?token=x"), DownloadCancelled("stop")])
async def test_worker_sanitizes_failure_or_cancellation(tmp_path: Path, outcome: Exception) -> None:
    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=snapshot())
    client._pull_job = AsyncMock(side_effect=outcome)
    value = manager(tmp_path, client=client, max_concurrent_downloads=1)
    job = await value.enqueue("example")
    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()
    expected = "cancelled" if isinstance(outcome, DownloadCancelled) else "failed"
    assert result.state == expected
    assert "token=x" not in (result.error_message or "")


@pytest.mark.asyncio
async def test_wait_calls_sync_and_async_observers(tmp_path: Path) -> None:
    value = manager(tmp_path)
    job = await value.enqueue("example")
    await value.cancel(job.id)
    sync_states = []
    assert (
        await value.wait(job.id, on_update=lambda item: sync_states.append(item.state))
    ).state == "cancelled"
    async_states = []

    async def observe(item):
        async_states.append(item.state)

    assert (await value.wait(job.id, on_update=observe)).state == "cancelled"
    assert sync_states == ["cancelled"] and async_states == ["cancelled"]


@pytest.mark.asyncio
async def test_heartbeat_detects_lost_claim(tmp_path: Path) -> None:
    value = manager(tmp_path)
    value.store.cancellation_requested = MagicMock(return_value=False)
    value.store.heartbeat = MagicMock(return_value=False)
    cancel = __import__("threading").Event()
    with pytest.MonkeyPatch.context() as monkeypatch:
        sleep = AsyncMock()
        monkeypatch.setattr(asyncio, "sleep", sleep)
        await value._heartbeat("job", "token", cancel)
    assert cancel.is_set()


def test_store_invalid_transitions_and_chunk_fencing(tmp_path: Path) -> None:
    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("example", str(tmp_path / "model"), 2)
    claimed = store.claim("worker", 60)
    assert claimed
    assert not store.prepare_chunks("missing", claimed[1], 1, 1)
    with pytest.raises(ValueError):
        store.prepare_chunks("missing", claimed[1], 1, 0)
    assert not store.record_progress("missing", claimed[1], {})
    assert not store.reset_file("missing", claimed[1])
    with pytest.raises(ValueError):
        store.finish(job.id, claimed[1], "downloading")
