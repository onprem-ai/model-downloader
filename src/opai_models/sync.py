"""Manifest-driven synchronization of completed model directories."""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opai_models.async_client import _file_sha256, _secure_parent, _write_bytes
from opai_models.client import ModelDownloadError
from opai_models.download import pull_file_with_state
from opai_models.metadata import parse_sha256sums
from opai_models.signatures import verify_sigstore_bundle

SyncProgress = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class SyncResult:
    model_id: str
    destination: Path
    files: int
    bytes: int
    reused_files: int
    downloaded_files: int
    deleted_files: int
    rehashed_files: int


def _files(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ModelDownloadError("model directory must not contain symbolic links")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            result[relative] = path
    return result


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP}:
            raise
        shutil.copy2(source, destination)


def _replace_directory(staging: Path, destination: Path, backup: Path) -> None:
    os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except BaseException:
        os.replace(backup, destination)
        raise
    shutil.rmtree(backup)


async def sync_model(
    client: Any,
    model_id: str,
    destination: Path,
    *,
    rehash: bool = False,
    delete: bool = False,
    chunk_size: int = 64 * 1024 * 1024,
    workers: int = 4,
    request_retries: int = 8,
    initial_backoff: float = 0.5,
    max_backoff: float = 60.0,
    progress: SyncProgress | None = None,
) -> SyncResult:
    """Synchronize a completed directory without relying on queue state."""
    destination = destination.expanduser().resolve()
    staging = destination.with_name(f".{destination.name}.sync.partial")
    backup = destination.with_name(f".{destination.name}.sync.previous")
    if not destination.exists() and backup.is_dir() and not backup.is_symlink():
        await asyncio.to_thread(os.replace, backup, destination)
    if not destination.is_dir() or destination.is_symlink():
        raise ModelDownloadError("sync destination must be an existing model directory")
    if backup.exists():
        raise ModelDownloadError("sync backup already exists; inspect or remove it")
    if staging.is_symlink():
        raise ModelDownloadError("sync staging path must not be a symbolic link")
    if staging.exists() and not staging.is_dir():
        raise ModelDownloadError("sync staging path must be a directory")
    snapshot = await client.snapshot_model(model_id)
    if snapshot.sha256sums is None or any(item.sha256 is None for item in snapshot.files):
        raise ModelDownloadError("sync requires the remote SHA256SUMS inventory")
    remote_manifest = snapshot.sha256sums.encode()
    remote_checksums = parse_sha256sums(remote_manifest)
    local_files = await asyncio.to_thread(_files, destination)
    local_manifest: bytes | None = None
    local_checksums: dict[str, str] = {}
    manifest_path = destination / "SHA256SUMS"
    try:
        local_manifest = await asyncio.to_thread(manifest_path.read_bytes)
        local_checksums = parse_sha256sums(local_manifest)
    except (OSError, ModelDownloadError):
        local_manifest = None
        local_checksums = {}
    manifests_match = local_manifest == remote_manifest

    staging.mkdir(mode=0o700, exist_ok=True)
    staging.chmod(0o700)
    staged_files = await asyncio.to_thread(_files, staging)
    reused = downloaded = deleted = rehashed_files = 0
    expected_paths = set(remote_checksums)
    try:
        for item in snapshot.files:
            target = _secure_parent(staging, item.relative_path)
            staged = staged_files.get(item.relative_path)
            local = local_files.get(item.relative_path)
            reusable: Path | None = None
            if staged is not None and staged.stat().st_size == item.size:
                rehashed_files += 1
                if await asyncio.to_thread(_file_sha256, staged) == item.sha256:
                    reusable = staged
            if reusable is None and local is not None and local.stat().st_size == item.size:
                if not rehash and (
                    manifests_match or local_checksums.get(item.relative_path) == item.sha256
                ):
                    reusable = local
                else:
                    rehashed_files += 1
                    if await asyncio.to_thread(_file_sha256, local) == item.sha256:
                        reusable = local
            if reusable is not None:
                if reusable != target:
                    target.unlink(missing_ok=True)
                    await asyncio.to_thread(_link_or_copy, reusable, target)
                reused += 1
                if progress:
                    progress({"event": "file_reused", "path": item.relative_path})
                continue

            target.unlink(missing_ok=True)
            completed_chunks: set[int] = set()

            current_path, current_size = item.relative_path, item.size

            def mark_complete(
                chunk: int,
                completed_bytes: int,
                bytes_per_second: int,
                path: str = current_path,
                total: int = current_size,
            ) -> None:
                if progress:
                    progress(
                        {
                            "event": "chunk_complete",
                            "path": path,
                            "chunk": chunk,
                            "completed_bytes": completed_bytes,
                            "total_bytes": total,
                            "bytes_per_second": bytes_per_second,
                        }
                    )

            await pull_file_with_state(
                client._client,
                snapshot.model_id,
                item.object_path,
                target,
                expected_source_id=item.source_id,
                expected_size=item.size,
                expected_sha256=item.sha256,
                completed_chunks=completed_chunks,
                mark_complete=mark_complete,
                chunk_size=chunk_size,
                workers=workers,
                request_retries=request_retries,
                initial_backoff=initial_backoff,
                max_backoff=max_backoff,
                verify_checksum=True,
            )
            downloaded += 1
            if progress:
                progress({"event": "file_downloaded", "path": item.relative_path})

        metadata = {"SHA256SUMS", "SHA256SUMS.sigstore.json"}
        extras = set(local_files) - expected_paths - metadata
        if delete:
            deleted = len(extras)
        else:
            for relative in sorted(extras):
                await asyncio.to_thread(
                    _link_or_copy,
                    local_files[relative],
                    _secure_parent(staging, relative),
                )

        await asyncio.to_thread(_write_bytes, staging / "SHA256SUMS", remote_manifest)
        if client.verify_signatures:
            signature = await client._client.read_small(
                snapshot.model_id,
                "SHA256SUMS.sigstore.json",
                maximum=16 * 1024 * 1024,
            )
            if client.sigstore_identity is None:
                raise ModelDownloadError("Sigstore trusted identity and issuer are required")
            await asyncio.to_thread(
                verify_sigstore_bundle,
                remote_manifest,
                signature,
                client.sigstore_identity,
                offline=client.sigstore_offline,
            )
            await asyncio.to_thread(_write_bytes, staging / "SHA256SUMS.sigstore.json", signature)
        elif manifests_match and "SHA256SUMS.sigstore.json" in local_files and not delete:
            await asyncio.to_thread(
                _link_or_copy,
                local_files["SHA256SUMS.sigstore.json"],
                staging / "SHA256SUMS.sigstore.json",
            )
        desired = expected_paths | {"SHA256SUMS"}
        if client.verify_signatures or (
            manifests_match and "SHA256SUMS.sigstore.json" in local_files and not delete
        ):
            desired.add("SHA256SUMS.sigstore.json")
        if not delete:
            desired.update(extras)
        for relative, path in (await asyncio.to_thread(_files, staging)).items():
            if relative not in desired:
                path.unlink()
        await asyncio.to_thread(_replace_directory, staging, destination, backup)
    except BaseException:
        raise

    return SyncResult(
        model_id=snapshot.model_id,
        destination=destination,
        files=snapshot.file_count,
        bytes=snapshot.total_bytes,
        reused_files=reused,
        downloaded_files=downloaded,
        deleted_files=deleted,
        rehashed_files=rehashed_files,
    )
