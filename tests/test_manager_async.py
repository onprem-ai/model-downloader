import asyncio
from contextlib import closing
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from opai_models.client import ModelDownloadError, TransientModelDownloadError
from opai_models.download import ChecksumMismatchError, DownloadCancelled
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
    assert (await value.get(job.id)).model_dir_name == "example"
    assert (await value.list(state="queued"))[0].id == job.id
    assert (await value.cancel(job.id)).state == "cancelled"
    with pytest.raises(Exception, match="not retryable"):
        await value.retry(job.id)
    async with value:
        await value.start()  # idempotent
    assert not value._tasks
    with pytest.raises(ValueError, match="inside"):
        await value.enqueue("example", "../escape")


def test_manager_rejects_bad_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        manager(tmp_path, max_concurrent_downloads=0)
    with pytest.raises(ValueError):
        manager(tmp_path, lease_seconds=29)
    with pytest.raises(ValueError):
        manager(tmp_path, poll_interval=0)
    with pytest.raises(ValueError):
        manager(tmp_path, request_retries=-1)
    with pytest.raises(ValueError):
        manager(tmp_path, integrity_retries=-1)
    with pytest.raises(ValueError):
        manager(tmp_path, initial_backoff=0)
    with pytest.raises(ValueError):
        manager(tmp_path, initial_backoff=2, max_backoff=1)
    with pytest.raises(ValueError):
        manager(tmp_path, no_progress_timeout=0)
    with pytest.raises(ValueError):
        manager(tmp_path, overall_timeout=-1)
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
async def test_transient_failure_automatically_retries_and_records_error(tmp_path: Path) -> None:
    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=snapshot())
    calls = 0

    async def pull(job_id, destination, store, token, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientModelDownloadError("temporary")
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

    client._pull_job = AsyncMock(side_effect=pull)
    value = manager(
        tmp_path,
        client=client,
        max_concurrent_downloads=1,
        initial_backoff=0.001,
        max_backoff=0.001,
    )
    job = await value.enqueue("example")
    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    errors = await value.errors(job.id)
    await value.close()
    assert result.state == "completed" and result.run_count == 2
    assert errors[0].retryable and errors[0].file_id is None
    assert errors[0].message == (
        "Temporary download failure (TransientModelDownloadError): temporary"
    )


@pytest.mark.asyncio
async def test_old_permanent_snapshot_failure_is_not_mislabeled_as_timeout(
    tmp_path: Path, monkeypatch
) -> None:
    import opai_models.manager as module

    client = MagicMock()
    client.snapshot_model = AsyncMock(side_effect=ModelDownloadError("remote model is empty"))
    client._pull_job = AsyncMock()
    value = manager(
        tmp_path,
        client=client,
        max_concurrent_downloads=1,
        no_progress_timeout=1,
    )
    job = await value.enqueue("example")
    future = module.datetime(2030, 1, 1, tzinfo=module.UTC)
    monkeypatch.setattr(module, "_now", lambda: future)

    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()

    assert result.state == "failed"
    assert result.error_code == "ModelDownloadError"
    assert result.error_message == "Download failed (ModelDownloadError): remote model is empty"


@pytest.mark.asyncio
async def test_no_progress_timeout_stops_retrying(tmp_path: Path, monkeypatch) -> None:
    import opai_models.manager as module

    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=snapshot())

    async def stalled(job_id, destination, store, token, **kwargs):
        del destination, token, kwargs
        with closing(store._connect()) as connection, connection:
            connection.execute(
                "UPDATE download_jobs SET last_progress_at=? WHERE id=?",
                ("2020-01-01T00:00:00+00:00", job_id),
            )
        raise TransientModelDownloadError("temporary")

    client._pull_job = AsyncMock(side_effect=stalled)
    value = manager(
        tmp_path,
        client=client,
        max_concurrent_downloads=1,
        no_progress_timeout=1,
    )
    job = await value.enqueue("example")
    future = module.datetime(2030, 1, 1, tzinfo=module.UTC)
    monkeypatch.setattr(module, "_now", lambda: future)
    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()
    assert result.state == "failed"
    assert result.error_code == "no_progress_timeout"
    assert result.error_message == (
        "Download made no progress before the configured timeout: temporary"
    )
    assert (await value.errors(job.id))[0].retryable is False


@pytest.mark.asyncio
async def test_overall_timeout_is_terminal(tmp_path: Path) -> None:
    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=snapshot())

    async def pull(*args, **kwargs):
        await asyncio.sleep(0.01)
        if kwargs["should_cancel"]():
            raise __import__(
                "opai_models.download", fromlist=["DownloadCancelled"]
            ).DownloadCancelled("deadline")

    client._pull_job = AsyncMock(side_effect=pull)
    value = manager(tmp_path, client=client, max_concurrent_downloads=1, overall_timeout=0.001)
    job = await value.enqueue("example")
    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()
    assert result.state == "failed" and result.error_code == "overall_timeout"


@pytest.mark.asyncio
async def test_explicit_active_cancel_is_terminal(tmp_path: Path) -> None:
    entered = asyncio.Event()

    async def pull(*args, **kwargs):
        entered.set()
        while not kwargs["should_cancel"]():
            await asyncio.sleep(0.001)
        raise DownloadCancelled("cancelled")

    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=snapshot())
    client._pull_job = AsyncMock(side_effect=pull)
    value = manager(tmp_path, client=client, max_concurrent_downloads=1)
    job = await value.enqueue("example")
    await value.start()
    await asyncio.wait_for(entered.wait(), 1)
    await value.cancel(job.id)
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()
    assert result.state == "cancelled"


@pytest.mark.asyncio
async def test_worker_sanitizes_permanent_failure(tmp_path: Path) -> None:
    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=snapshot())
    client._pull_job = AsyncMock(
        side_effect=ModelDownloadError(
            "License Server denied https://storage.example/model?token=secret-value"
        )
    )
    value = manager(tmp_path, client=client, max_concurrent_downloads=1)
    job = await value.enqueue("example")
    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()
    assert result.state == "failed"
    assert result.error_message == (
        "Download failed (ModelDownloadError): License Server denied [URL REDACTED]"
    )
    errors = await value.errors(job.id)
    assert errors and errors[0].retryable is False
    assert errors[0].message == result.error_message


@pytest.mark.asyncio
async def test_checksum_failure_stops_after_integrity_budget(tmp_path: Path) -> None:
    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=snapshot())

    async def corrupt(job_id, destination, store, token, **kwargs):
        file = store.files(job_id)[0]
        store.reset_file(file.id, token, integrity_failure=True)
        raise ChecksumMismatchError("bad")

    client._pull_job = AsyncMock(side_effect=corrupt)
    value = manager(
        tmp_path,
        client=client,
        max_concurrent_downloads=1,
        integrity_retries=1,
        initial_backoff=0.001,
        max_backoff=0.001,
    )
    job = await value.enqueue("example")
    await value.start()
    result = await asyncio.wait_for(value.wait(job.id, poll_interval=0.01), 2)
    await value.close()
    assert result.state == "failed"
    assert result.error_code == "integrity_retries_exhausted"
    assert result.error_message == ("Download failed repeated integrity verification: bad")
    assert result.run_count == 2


@pytest.mark.asyncio
async def test_wait_calls_sync_and_async_observers(tmp_path: Path) -> None:
    value = manager(tmp_path)
    job = await value.enqueue("example")
    await value.cancel(job.id)
    states = []

    async def observe(item):
        states.append(item.state)

    assert (await value.wait(job.id, on_update=observe)).state == "cancelled"
    assert states == ["cancelled"]


@pytest.mark.asyncio
async def test_wait_rejects_synchronous_observer(tmp_path: Path) -> None:
    value = manager(tmp_path)
    job = await value.enqueue("example")
    with pytest.raises(TypeError, match="on_update must be an async callable"):
        await value.wait(job.id, on_update=lambda item: None)


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
    job = store.enqueue("example", str(tmp_path / "model"))
    claimed = store.claim("worker", 60)
    assert claimed
    assert not store.prepare_chunks("missing", claimed[1], 1, 1)
    with pytest.raises(ValueError):
        store.prepare_chunks("missing", claimed[1], 1, 0)
    assert not store.record_progress("missing", claimed[1], {})
    assert not store.reset_file("missing", claimed[1])
    with pytest.raises(ValueError):
        store.finish(job.id, claimed[1], "downloading")
