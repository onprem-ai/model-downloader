"""Async programmatic API for model discovery and downloads."""

import asyncio
import hashlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from opai_models.client import LicenseClient, ModelAccess, ModelDownloadError
from opai_models.download import (
    CancelCallback,
    ChecksumMismatchError,
    DownloadCancelled,
    pull_file_with_state,
)
from opai_models.metadata import SourceDocument, parse_sha256sums, parse_source, read_source
from opai_models.signatures import SigstoreIdentity, verify_sigstore_bundle
from opai_models.snapshot import ModelSnapshot, snapshot_model

LicenseProvider = Callable[[], str]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(path: Path, expected: str | None) -> bool:
    return expected is not None and path.is_file() and _file_sha256(path) == expected


def _completed_directory_matches(
    destination: Path,
    job: Any,
    files: list[Any],
    source: SourceDocument,
    *,
    verify_checksums: bool,
    verify_signatures: bool,
    trusted_identity: SigstoreIdentity | None,
    sigstore_offline: bool,
) -> bool:
    try:
        if read_source(destination / ".source.json") != source:
            return False
        expected_paths = {item.relative_path for item in files}
        sums_path = destination / "SHA256SUMS"
        if verify_checksums or verify_signatures or sums_path.is_file():
            sums = sums_path.read_bytes()
            if "sha256:" + hashlib.sha256(sums).hexdigest() != job.snapshot_sha256:
                return False
            parsed = parse_sha256sums(sums)
            if parsed != {item.relative_path: item.expected_sha256 for item in files}:
                return False
            expected_paths.add("SHA256SUMS")
        else:
            sums = b""
        signature_path = destination / "SHA256SUMS.sigstore.json"
        if verify_signatures:
            if trusted_identity is None or not signature_path.is_file():
                return False
            verify_sigstore_bundle(
                sums,
                signature_path.read_bytes(),
                trusted_identity,
                offline=sigstore_offline,
            )
            expected_paths.add("SHA256SUMS.sigstore.json")
        actual_paths = {
            path.relative_to(destination).as_posix()
            for path in destination.rglob("*")
            if path.is_file()
        }
        return actual_paths == expected_paths and (
            not verify_checksums
            or all(
                _matches(destination / item.relative_path, item.expected_sha256) for item in files
            )
        )
    except (OSError, ModelDownloadError):
        return False


def _secure_parent(root: Path, relative_path: str) -> Path:
    current = root
    for part in Path(relative_path).parent.parts:
        current = current / part
        if current.is_symlink():
            raise ModelDownloadError("staging path must not contain symbolic links")
        current.mkdir(mode=0o700, exist_ok=True)
        current.chmod(0o700)
    return root / relative_path


def _write_bytes(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


class AsyncModelClient:
    """Reusable async facade over the dependency-free transfer implementation.

    Blocking filesystem and urllib operations run in worker threads, so callers
    never block an asyncio/FastAPI event loop. Credentials are requested for
    each operation and are never retained in durable state.
    """

    def __init__(
        self,
        api_url: str,
        license_provider: LicenseProvider,
        *,
        timeout: float = 30,
        verify_checksums: bool = True,
        verify_signatures: bool = True,
        sigstore_identity: str | None = None,
        sigstore_issuer: str | None = None,
        sigstore_offline: bool = False,
    ) -> None:
        if verify_signatures and (not sigstore_identity or not sigstore_issuer):
            raise ValueError(
                "sigstore_identity and sigstore_issuer are required unless "
                "signature verification is disabled"
            )
        self.api_url = api_url
        self.license_provider = license_provider
        self.timeout = timeout
        self.verify_checksums = verify_checksums
        self.verify_signatures = verify_signatures
        self.sigstore_identity = (
            SigstoreIdentity(sigstore_identity, sigstore_issuer)
            if sigstore_identity and sigstore_issuer
            else None
        )
        self.sigstore_offline = sigstore_offline

    def _client(self) -> LicenseClient:
        return LicenseClient(self.api_url, self.license_provider(), self.timeout)

    async def list_models(self, *, limit: int = 1000) -> dict[str, Any]:
        return await asyncio.to_thread(self._client().list_models, limit=limit)

    async def get_model_file(self, model_id: str, relative_path: str) -> ModelAccess:
        return await asyncio.to_thread(self._client().access, model_id, relative_path)

    async def snapshot_model(self, model_id: str) -> ModelSnapshot:
        return await asyncio.to_thread(
            snapshot_model,
            self._client(),
            model_id,
            verify_checksums=self.verify_checksums,
            verify_signatures=self.verify_signatures,
            trusted_identity=self.sigstore_identity,
            sigstore_offline=self.sigstore_offline,
        )

    async def get_source(self, model_id: str) -> SourceDocument:
        """Fetch and validate provenance without retaining access material."""
        model = LicenseClient._model_id(model_id)
        data = await asyncio.to_thread(self._client().read_small, model, ".source.json")
        return parse_source(data)

    async def _pull_job(
        self,
        job_id: str,
        destination: Path,
        store: Any,
        claim_token: str,
        *,
        chunk_size: int = 64 * 1024 * 1024,
        workers: int = 4,
        request_retries: int = 8,
        timeout: float = 60,
        initial_backoff: float = 0.5,
        max_backoff: float = 60.0,
        should_cancel: CancelCallback | None = None,
    ) -> Path:
        """Download a persisted immutable model snapshot and publish it atomically."""
        destination = destination.expanduser().resolve()
        job = await asyncio.to_thread(store.get, job_id)
        files = await asyncio.to_thread(store.files, job_id)
        source = await asyncio.to_thread(store.source, job_id)
        if destination.exists():
            if await asyncio.to_thread(
                _completed_directory_matches,
                destination,
                job,
                files,
                source,
                verify_checksums=self.verify_checksums,
                verify_signatures=self.verify_signatures,
                trusted_identity=self.sigstore_identity,
                sigstore_offline=self.sigstore_offline,
            ):
                for item in files:
                    if item.state != "completed" and not await asyncio.to_thread(
                        store.update_file,
                        item.id,
                        claim_token,
                        completed_bytes=item.expected_size,
                        state="completed",
                        sha256=item.expected_sha256,
                    ):
                        raise DownloadCancelled("download lease was lost")
                return destination
            raise ModelDownloadError("existing destination does not match model snapshot")
        staging = destination.with_name(f".{destination.name}.{job_id}.partial")
        if staging.is_symlink():
            raise ModelDownloadError("staging path must not be a symbolic link")
        staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging.chmod(0o700)
        if not files or not job.snapshot_sha256:
            raise ModelDownloadError("job has no immutable model snapshot")

        semaphore = asyncio.Semaphore(workers)

        async def transfer(item: Any) -> None:
            if item.state == "completed":
                candidate = staging / item.relative_path
                reusable = candidate.is_file() and (
                    candidate.stat().st_size == item.expected_size
                    if not self.verify_checksums
                    else await asyncio.to_thread(_matches, candidate, item.expected_sha256)
                )
                if reusable:
                    return
                if not await asyncio.to_thread(store.reset_file, item.id, claim_token):
                    raise DownloadCancelled("download lease was lost")
            target = await asyncio.to_thread(_secure_parent, staging, item.relative_path)
            if target.is_symlink():
                raise ModelDownloadError("staging file must not be a symbolic link")
            await asyncio.to_thread(
                store.prepare_chunks, item.id, claim_token, item.expected_size, chunk_size
            )

            completed = await asyncio.to_thread(store.completed_chunks, item.id)
            if completed and (not target.is_file() or target.stat().st_size != item.expected_size):
                if not await asyncio.to_thread(store.reset_file, item.id, claim_token):
                    raise DownloadCancelled("download lease was lost")
                completed = set()

            def mark_complete(chunk: int, completed_bytes: int, bytes_per_second: int) -> None:
                updated = store.record_progress(
                    item.id,
                    claim_token,
                    {
                        "event": "chunk_complete",
                        "chunk": chunk,
                        "completed_bytes": completed_bytes,
                        "bytes_per_second": bytes_per_second,
                    },
                )
                if not updated:
                    raise DownloadCancelled("download lease was lost")

            def record_error(event: dict[str, Any]) -> None:
                if not store.record_error(item.id, claim_token, event):
                    raise DownloadCancelled("download lease was lost")

            try:
                async with semaphore:
                    await asyncio.to_thread(
                        pull_file_with_state,
                        self._client(),
                        job.model_id,
                        item.object_path,
                        target,
                        expected_source_id=item.source_id,
                        expected_size=item.expected_size,
                        expected_sha256=item.expected_sha256,
                        completed_chunks=completed,
                        mark_complete=mark_complete,
                        chunk_size=chunk_size,
                        workers=workers,
                        request_retries=request_retries,
                        timeout=timeout,
                        initial_backoff=initial_backoff,
                        max_backoff=max_backoff,
                        should_cancel=should_cancel,
                        on_error=record_error,
                        verify_checksum=self.verify_checksums,
                    )
            except ChecksumMismatchError as exc:
                await asyncio.to_thread(
                    store.record_error,
                    item.id,
                    claim_token,
                    {
                        "error_type": type(exc).__name__,
                        "message": "Final SHA-256 verification failed",
                        "retryable": True,
                    },
                )
                await asyncio.to_thread(
                    store.reset_file,
                    item.id,
                    claim_token,
                    integrity_failure=True,
                )
                raise
            digest = (
                await asyncio.to_thread(_file_sha256, target)
                if self.verify_checksums
                else item.expected_sha256
            )
            if self.verify_checksums and digest != item.expected_sha256:
                await asyncio.to_thread(
                    store.record_error,
                    item.id,
                    claim_token,
                    {
                        "error_type": "ChecksumMismatchError",
                        "message": "Final SHA-256 verification failed",
                        "retryable": True,
                    },
                )
                await asyncio.to_thread(
                    store.reset_file,
                    item.id,
                    claim_token,
                    integrity_failure=True,
                )
                raise ChecksumMismatchError(
                    f"final SHA-256 verification failed for {item.relative_path}"
                )
            if not await asyncio.to_thread(
                store.update_file,
                item.id,
                claim_token,
                completed_bytes=item.expected_size,
                state="completed",
                sha256=digest,
            ):
                raise DownloadCancelled("download lease was lost")

        for item in files:
            await transfer(item)
        if not await asyncio.to_thread(store.mark_verifying, job_id, claim_token):
            raise DownloadCancelled("download lease was lost")
        sums: str | None = None
        if self.verify_checksums or self.verify_signatures:
            if any(item.expected_sha256 is None for item in files):
                raise ModelDownloadError("persisted model snapshot lacks checksums")
            sums = "".join(f"{item.expected_sha256}  {item.relative_path}\n" for item in files)
            if "sha256:" + hashlib.sha256(sums.encode()).hexdigest() != job.snapshot_sha256:
                raise ModelDownloadError("persisted model snapshot is inconsistent")
            await asyncio.to_thread(_write_bytes, staging / "SHA256SUMS", sums.encode())
        if self.verify_signatures:
            if sums is None:
                raise ModelDownloadError("signature verification requires SHA256SUMS")
            signature = await asyncio.to_thread(
                self._client().read_small,
                job.model_id,
                "SHA256SUMS.sigstore.json",
                maximum=16 * 1024 * 1024,
            )
            if self.sigstore_identity is None:
                raise ModelDownloadError("Sigstore trusted identity and issuer are required")
            await asyncio.to_thread(
                verify_sigstore_bundle,
                sums.encode(),
                signature,
                self.sigstore_identity,
                offline=self.sigstore_offline,
            )
            await asyncio.to_thread(
                _write_bytes,
                staging / "SHA256SUMS.sigstore.json",
                signature,
            )
        if await asyncio.to_thread(store.cancellation_requested, job_id, claim_token):
            raise DownloadCancelled("download lease was lost")
        await asyncio.to_thread(os.replace, staging, destination)
        return destination

    async def __aenter__(self) -> "AsyncModelClient":
        return self

    async def __aexit__(self, *args: object) -> None:
        return None
