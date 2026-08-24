"""Minimal FastAPI integration using one process-scoped DownloadManager.

Set OPAI_LICENSE_KEY, OPAI_SIGSTORE_IDENTITY, and OPAI_SIGSTORE_ISSUER in the
server environment. Never accept or return the license key through these routes.
"""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from opai_models import DownloadManager
from opai_models.async_client import AsyncModelClient
from opai_models.manager import JobNotFoundError


class AppState(TypedDict):
    downloads: DownloadManager


class DownloadRequest(BaseModel):
    model_id: str
    destination: str | None = None


def required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[AppState]:
    # The callback supplies the credential only when making an API request. The
    # manager never stores it in SQLite.
    client = AsyncModelClient(
        "https://license.api.onprem.ai",
        license_provider=lambda: required_environment("OPAI_LICENSE_KEY"),
        sigstore_identity=required_environment("OPAI_SIGSTORE_IDENTITY"),
        sigstore_issuer=required_environment("OPAI_SIGSTORE_ISSUER"),
    )
    manager = DownloadManager(
        database_path=Path("/var/lib/app/model-downloads.sqlite"),
        download_directory=Path("/var/lib/app/models"),
        client=client,
        max_concurrent_downloads=1,
    )
    await manager.start()
    try:
        # FastAPI copies this mapping into app.state for the lifespan.
        yield {"downloads": manager}
    finally:
        await manager.close()


app = FastAPI(lifespan=lifespan)


def get_download_manager(request: Request) -> DownloadManager:
    return request.state.downloads


Downloads = Annotated[DownloadManager, Depends(get_download_manager)]


@app.post("/downloads", status_code=202)
async def create_download(body: DownloadRequest, downloads: Downloads) -> dict[str, object]:
    job = await downloads.enqueue(body.model_id, body.destination)
    return job.to_dict()


@app.get("/downloads/{job_id}")
async def get_download(job_id: str, downloads: Downloads) -> dict[str, object]:
    try:
        return (await downloads.get(job_id)).to_dict()
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="download not found") from None


@app.delete("/downloads/{job_id}")
async def cancel_download(job_id: str, downloads: Downloads) -> dict[str, object]:
    try:
        return (await downloads.cancel(job_id)).to_dict()
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail="download not found") from None
