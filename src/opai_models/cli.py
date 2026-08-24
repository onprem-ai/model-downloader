"""Command-line interface for discovering and downloading OnPrem AI models."""

import argparse
import asyncio
import getpass
import json
import os
import sys
from pathlib import Path
from typing import Any

from opai_models.async_client import AsyncModelClient
from opai_models.client import LicenseClient, ModelDownloadError
from opai_models.manager import DownloadManager

DEFAULT_API_URL = "https://license.api.onprem.ai"
DEFAULT_LICENSE_ENV = "OPAI_LICENSE_KEY"
DEFAULT_SIGSTORE_IDENTITY_ENV = "OPAI_SIGSTORE_IDENTITY"
DEFAULT_SIGSTORE_ISSUER_ENV = "OPAI_SIGSTORE_ISSUER"


def _license_key(environment_name: str, *, prompt: bool) -> str:
    value = os.environ.get(environment_name, "").strip()
    if value:
        return value
    if prompt and sys.stdin.isatty():
        return getpass.getpass("OnPrem AI license key: ").strip()
    raise ModelDownloadError(
        f"license key unavailable; set {environment_name} or run interactively"
    )


def _human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    raise AssertionError("unreachable")


def _progress(value: dict[str, Any]) -> None:
    if value["event"] == "chunk_complete":
        completed = int(value["completed_bytes"])
        total = int(value["total_bytes"])
        percentage = completed * 100 / max(total, 1)
        speed = _human_size(int(value["bytes_per_second"])) + "/s"
        print(
            f"\r{percentage:6.2f}%  {_human_size(completed)} / {_human_size(total)}  {speed}",
            end="",
            file=sys.stderr,
            flush=True,
        )
    elif value["event"] == "complete":
        print(file=sys.stderr)
        print(f"Downloaded {value['path']} -> {value['destination']}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="opai-models",
        description="List and securely download models from OnPrem AI storage.",
    )
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--license-env", default=DEFAULT_LICENSE_ENV)
    parser.add_argument("--no-prompt", action="store_true")
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list available model IDs")
    list_parser.add_argument("--limit", type=int, default=1000)

    info_parser = subparsers.add_parser("info", help="show metadata for one model file")
    info_parser.add_argument("model_id")
    info_parser.add_argument("relative_path")

    pull_parser = subparsers.add_parser("pull", help="download or resume one model directory")
    pull_parser.add_argument("model_id")
    pull_parser.add_argument("destination", nargs="?", type=Path)
    pull_parser.add_argument("--download-directory", type=Path, default=Path.cwd())
    pull_parser.add_argument("--database", type=Path, default=Path(".opai-model-downloads.sqlite"))
    pull_parser.add_argument("--chunk-size", type=int, default=64 * 1024 * 1024)
    pull_parser.add_argument("--workers", type=int, default=4)
    pull_parser.add_argument("--request-retries", type=int, default=8)
    pull_parser.add_argument("--integrity-retries", type=int, default=2)
    pull_parser.add_argument("--initial-backoff", type=float, default=0.5)
    pull_parser.add_argument("--max-backoff", type=float, default=60.0)
    pull_parser.add_argument("--no-progress-timeout", type=float, default=3600.0)
    pull_parser.add_argument(
        "--overall-timeout",
        type=float,
        default=0.0,
        help="maximum total seconds, or 0 for unlimited",
    )
    pull_parser.add_argument(
        "--sigstore-identity",
        default=os.environ.get(DEFAULT_SIGSTORE_IDENTITY_ENV),
        help=f"trusted signer identity (or {DEFAULT_SIGSTORE_IDENTITY_ENV})",
    )
    pull_parser.add_argument(
        "--sigstore-issuer",
        default=os.environ.get(DEFAULT_SIGSTORE_ISSUER_ENV),
        help=f"trusted signer OIDC issuer (or {DEFAULT_SIGSTORE_ISSUER_ENV})",
    )
    pull_parser.add_argument(
        "--sigstore-offline",
        action="store_true",
        help="use cached Sigstore trust roots without refreshing them",
    )
    pull_parser.add_argument(
        "--skip-checksum-verification",
        action="store_true",
        help="download without comparing file contents to SHA256SUMS",
    )
    pull_parser.add_argument(
        "--skip-signature-verification",
        action="store_true",
        help="allow an unsigned or unverified SHA256SUMS manifest",
    )
    return parser


async def _pull(args: argparse.Namespace, license_provider) -> int:
    client = AsyncModelClient(
        args.api_url,
        license_provider,
        verify_checksums=not args.skip_checksum_verification,
        verify_signatures=not args.skip_signature_verification,
        sigstore_identity=args.sigstore_identity,
        sigstore_issuer=args.sigstore_issuer,
        sigstore_offline=args.sigstore_offline,
    )
    manager = DownloadManager(
        args.database,
        args.download_directory,
        client,
        max_concurrent_downloads=1,
        chunk_size=args.chunk_size,
        range_workers=args.workers,
        request_retries=args.request_retries,
        integrity_retries=args.integrity_retries,
        initial_backoff=args.initial_backoff,
        max_backoff=args.max_backoff,
        no_progress_timeout=args.no_progress_timeout,
        overall_timeout=args.overall_timeout,
    )
    try:
        # Enqueue before starting workers so a duplicate destination cannot
        # accidentally start or cancel an already queued job in this process.
        queued = await manager.enqueue(args.model_id, args.destination)
        await manager.start()

        def show(job) -> None:
            if args.json:
                print(
                    json.dumps({"event": "progress", "job": job.to_dict()}, separators=(",", ":"))
                )
                return
            total = job.total_bytes or 0
            percentage = job.completed_bytes * 100 / max(total, 1)
            files = f"{job.completed_files}/{job.total_files or '?'} files"
            rate = f" {_human_size(job.bytes_per_second)}/s" if job.bytes_per_second else ""
            print(
                f"\r{job.state:12} {percentage:6.2f}% {files} "
                f"{_human_size(job.completed_bytes)}/{_human_size(total)}{rate}",
                end="",
                file=sys.stderr,
                flush=True,
            )

        result = await manager.wait(queued.id, on_update=show)
    finally:
        await manager.close()
    if result.state != "completed":
        raise ModelDownloadError(result.error_message or f"download {result.state}")
    event = {"event": "complete", "job": result.to_dict()}
    if args.json:
        print(json.dumps(event, separators=(",", ":")))
    else:
        print(file=sys.stderr)
        print(
            f"Downloaded {result.completed_files} files ({_human_size(result.completed_bytes)}) "
            f"to {result.destination}",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "limit", 1) not in range(1, 1001):
        parser.error("limit must be 1..1000")
    try:
        key = _license_key(args.license_env, prompt=not args.no_prompt)
        if args.command == "pull":
            return asyncio.run(_pull(args, lambda: key))
        client = LicenseClient(args.api_url, key)
        if args.command == "list":
            result = client.list_models(limit=args.limit)
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                for model_id in result["models"]:
                    print(model_id)
        else:
            access = client.access(args.model_id, args.relative_path)
            result = {
                "path": access.path,
                "size": access.size,
                "human_size": _human_size(access.size),
                "source_id": access.source_id,
                "expires_at": access.expires_at,
                "checksums": access.checksums,
                "range_supported": True,
            }
            print(json.dumps(result, indent=2))
    except (ModelDownloadError, OSError, ValueError, json.JSONDecodeError) as exc:
        if args.json:
            print(json.dumps({"event": "error", "error": str(exc)}), file=sys.stderr)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0
