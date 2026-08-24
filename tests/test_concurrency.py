import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from opai_models.download import DownloadCancelled
from opai_models.manager import DownloadManager, SQLiteQueueStore
from opai_models.metadata import SourceDocument
from opai_models.snapshot import ModelFile, ModelSnapshot


def persisted_snapshot() -> ModelSnapshot:
    provenance = SourceDocument.from_dict(
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
        provenance,
    )


def test_concurrent_stores_claim_job_once(tmp_path: Path) -> None:
    path = tmp_path / "queue.sqlite"
    SQLiteQueueStore(path).enqueue("example", str(tmp_path / "model"), 3)
    barrier = threading.Barrier(2)

    def claim(number: int):
        store = SQLiteQueueStore(path)
        barrier.wait()
        return store.claim(f"worker-{number}", 60)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, range(2)))
    assert sum(result is not None for result in results) == 1


def test_expired_exhausted_job_becomes_failed(tmp_path: Path, monkeypatch) -> None:
    import opai_models.manager as module

    store = SQLiteQueueStore(tmp_path / "queue.sqlite")
    job = store.enqueue("example", str(tmp_path / "model"), 1)
    current = [datetime(2026, 1, 1, tzinfo=UTC)]
    monkeypatch.setattr(module, "_now", lambda: current[0])
    assert store.claim("worker", 30)
    current[0] = datetime(2026, 1, 1, 0, 1, tzinfo=UTC)
    assert store.claim("other", 30) is None
    failed = store.get(job.id)
    assert failed.state == "failed"
    assert failed.error_code == "attempts_exhausted"


@pytest.mark.asyncio
async def test_graceful_close_leaves_active_job_reclaimable(tmp_path: Path) -> None:
    entered = __import__("asyncio").Event()

    async def pull(*args, **kwargs):
        entered.set()
        while not kwargs["should_cancel"]():
            await __import__("asyncio").sleep(0.01)
        raise DownloadCancelled("shutdown")

    client = MagicMock()
    client.snapshot_model = AsyncMock(return_value=persisted_snapshot())
    client._pull_job = AsyncMock(side_effect=pull)
    manager = DownloadManager(
        tmp_path / "queue.sqlite",
        tmp_path / "models",
        client,
        max_concurrent_downloads=1,
        lease_seconds=30,
        poll_interval=0.01,
    )
    job = await manager.enqueue("example")
    await manager.start()
    await __import__("asyncio").wait_for(entered.wait(), 1)
    await manager.close()
    assert (await manager.get(job.id)).state == "downloading"
