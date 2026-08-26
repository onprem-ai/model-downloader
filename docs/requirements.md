# Model Downloader Requirements

## 1. Identity and distribution

- Local repository name: `model-downloader`.
- Intended future GitHub repository: `onprem-ai/model-downloader`.
- Python distribution name: `opai-models`.
- Console command: `opai-models`.
- Import package: `opai_models`.
- Creating the public GitHub repository and publishing to PyPI are separate
  release actions requiring explicit approval.
- Future PyPI publication should use Trusted Publishing with OIDC, not a stored
  PyPI API token, and should publish a wheel, source distribution, hashes, and
  provenance from an immutable version tag.

## 2. Product scope

`opai-models` is specifically a model-directory downloader. It is not a general
artifact client and must not manage OCI images, Zarf packages, publisher
credentials, licenses, or License Server administration.

A model directory is identified publicly by one opaque model directory name.
The License Server maps that name to its configured storage namespace. For
example, model directory name `example-model` may contain:

```text
example-model/
├── config.json
├── tokenizer.json
├── model-00001-of-00004.safetensors
└── model-00002-of-00004.safetensors
```

One queued job represents the complete model directory, not an individual file.
The model is usable only after every file in the snapshotted directory has been
downloaded and verified.

## 3. License Server protocol

The client uses the model-specific License Server APIs:

```text
GET /v1/models
GET /v1/models/{model_dir_name}/files
GET /v1/models/{model_dir_name}/access/{relative_path:path}
```

Requirements:

- Accept only one-segment model directory names and normalized model-relative
  file paths.
- Never expose or require the backing S3 namespace in the downloader API.
- Recursively traverse paginated directory listings.
- Treat the access response as authoritative for object size, source identity,
  checksums, required request headers, and renewable signed URL.
- Download object bytes directly from S3, not through the License Server.
- Refresh an expired signed URL and require the refreshed `source_id` to remain
  unchanged.
- Never log or persist a license key or signed URL.

## 4. Public programmatic API

All supported programmatic APIs must be asynchronous and framework-neutral.
The primary public objects are:

```python
AsyncModelClient
DownloadManager
DownloadJob
ModelSnapshot
ModelFile
```

Required discovery methods:

```python
await client.list_models()
await client.get_model_file("example", "config.json")
await client.get_source("example")
await client.snapshot_model("example")
```

Durable model-directory transfer is owned by `DownloadManager`; low-level file
and range transfer functions are implementation details rather than public APIs.

Required queue methods:

```python
await manager.start()
await manager.close()
await manager.enqueue("example", destination="example")
await manager.get(job_id)
await manager.list(...)
await manager.errors(job_id)
await manager.cancel(job_id)
await manager.retry(job_id)
await manager.dismiss(job_id)
await manager.wait(job_id)
```

Rules:

- The public package API exposes `AsyncModelClient` and `DownloadManager`; the
  lower-level HTTP transport and synchronous SQLite implementation are internal.
- Methods return immutable typed records or plain serializable values; they must
  not depend on FastAPI/Pydantic response types.
- Blocking filesystem and SQLite work must not block the caller's event loop.
- One long-lived client should be reusable across requests and downloads.
- The license is supplied by an async callback for each authenticated API
  request, awaited on the event loop, and kept only in memory. It must never be
  stored in queue rows or metadata files.
- License Server and direct-download HTTP requests use a shared native-async
  HTTPX client and connection pool; network I/O must not use worker threads.
- Public progress callbacks are async-only and awaited on the event loop. Each
  callback completes before the corresponding operation continues. Callers must
  explicitly offload any blocking callback work.
- Potentially blocking filesystem calls, SQLite operations, hashing, and
  signature verification run outside the event loop. Durable chunk progress is
  recorded only after the range write and `fsync` complete.

## 5. CLI behavior

The CLI is a thin foreground consumer of the same public library:

```text
opai-models list
opai-models info <model-dir-name> <relative-file-path>
opai-models pull <model-dir-name> [destination-directory]
```

Requirements:

- `pull` enqueues one model-directory job, starts a local manager, and watches it
  until a terminal state.
- Show aggregate model progress: files verified, total files, completed bytes,
  expected bytes, transfer rate, run count, and state.
- `Ctrl-C` stops the foreground process without marking the durable job as a
  user cancellation; graceful shutdown releases its lease immediately so another
  manager can resume valid partial data. Explicit `manager.cancel(job_id)`
  performs cooperative cancellation.
- `--json` emits stable machine-readable events and errors.
- Interactive license entry uses a no-echo prompt.
- Noninteractive license input uses a configurable environment variable; the
  default is `OPAI_LICENSE_KEY`.
- The default API is `https://license.api.onprem.ai`.
- Signed URLs, credentials, and authorization headers must never be printed.

## 6. Model snapshot

Before downloading bytes, a worker recursively inventories the complete model
prefix and creates an immutable snapshot.

For every file, collect and persist:

- remote object path;
- relative path within the model directory;
- expected size in bytes;
- server-provided `source_id`;
- expected SHA-256 from the model's authoritative `SHA256SUMS` when checksum or
  signature verification is enabled;
- provider SHA-256 as an additional consistency check when available;
- ETag when available;
- object version ID when available.

At job level, persist:

- model directory name;
- total file count;
- total expected bytes;
- deterministic snapshot SHA-256.

When checksum or signature verification is enabled, the snapshot hash is
SHA-256 over the exact canonical `SHA256SUMS` bytes. With both checks explicitly
disabled, it is instead derived from the authenticated file paths, sizes, and
source identities. It never includes credentials, signed URLs, timestamps,
source provenance, local destinations, or transfer state.

After snapshot creation, retries use that exact snapshot. If current access
metadata differs in source identity, size, or an expected checksum, stop with a
source-changed error rather than silently changing the job.

Empty model directories are rejected.

## 7. Durable queue schema

The persistence model must represent the hierarchy directly.

### `download_jobs`

One row per model directory:

- UUID job ID;
- model directory name and final destination directory;
- state;
- total/verified file counts;
- expected/completed bytes;
- current transfer rate;
- restart count, consecutive failures, next retry time, and last progress time;
- snapshot SHA-256;
- sanitized error code/message;
- created, updated, started, and completed timestamps;
- worker ID, opaque claim/fencing token, lease expiry, and heartbeat time.

### `download_files`

One row per snapshotted object:

- file UUID and job ID;
- remote object path and safe relative path;
- expected size and completed bytes;
- source ID, expected SHA-256, ETag, and version ID;
- locally computed SHA-256;
- state and sanitized error;
- timestamps.

### `download_chunks`

One row per file range:

- file ID and chunk index;
- start and inclusive end byte;
- completion state and completion timestamp.

### `download_errors`

Append-only sanitized failure history:

- job ID and timestamp;
- optional file ID and chunk index, linked by foreign key to `download_chunks`;
- safe error type and message;
- retryable classification, HTTP status, request attempt, and backoff duration.

No error row may contain a signed URL, license key, authorization header, or
credential. The database is disposable: an incompatible database must be
deleted and recreated.

Constraints and indexes must enforce valid states, nonnegative sizes, unique
`(job_id, relative_path)`, unique `(file_id, chunk_index)`, and efficient claim,
lease, and status queries.

No table may contain license keys, signed URLs, access tokens, or registry
credentials.

## 8. Job and file state machines

Job states:

```text
queued
snapshotting
downloading
retry_wait
verifying
completed
failed
cancel_requested
cancelled
```

File states:

```text
queued
downloading
verifying
completed
failed
```

Rules:

- A job becomes `completed` only after every file is verified and the final
  directory is atomically installed.
- A queued cancellation becomes `cancelled` without running.
- An active cancellation becomes `cancel_requested`; workers stop cooperatively
  between blocks/chunks and finalize it as `cancelled`.
- Transient failures enter `retry_wait` and are resumed automatically after
  exponential full-jitter backoff.
- Every completed chunk resets consecutive failures and the no-progress clock.
- The overall job duration is unlimited by default; a job fails only after its
  configurable no-progress timeout or a permanent error.
- Integrity failures use a separate bounded retry budget.
- Failures retain verified files and safe partials for retry.
- Retry only downloads failed or incomplete files and never redownloads files
  whose local size and computed checksum still match the snapshot.
- A failed job can be explicitly retried; transient failures retry automatically while progress remains within the configured timeout.
- Terminal jobs can be dismissed from durable history. Active jobs must be cancelled before dismissal.
- Re-enqueuing the same model directory name and destination attaches to the
  active job, enabling a restarted CLI to resume it. A different model directory
  targeting the same final destination or staging tree is rejected.

## 9. SQLite claim, lease, and recovery rules

SQLite is the initial backend and supports multiple manager instances/processes
on one host sharing one local database file.

- Use WAL mode and `synchronous=FULL`.
- Use a bounded `busy_timeout`.
- Never place the SQLite database or its WAL files on NFS/network storage.
- Claim under `BEGIN IMMEDIATE`; select and transition the oldest eligible job
  while holding SQLite's writer lock.
- Assign a random claim/fencing token and lease expiry on every claim.
- Increment the informational run count atomically with claim.
- Every heartbeat, progress update, completion, failure, and cancellation
  finalization must match both job ID and current claim token.
- A stale worker whose lease was reclaimed must be unable to mutate the job.
- Workers heartbeat substantially faster than lease expiry.
- Expired `snapshotting`, `downloading`, or `verifying` jobs are reclaimable
  without consuming a finite lifetime retry budget.
- A cancellation held by a dead worker becomes `cancelled` after lease expiry.
- Polling is the reliable wake-up mechanism; in-process events are only a latency
  optimization.
- Claims must remain short transactions; never hold a database transaction open
  during network or filesystem work.

This is at-least-once execution with idempotent, resumable file writes—not a
claim of exactly-once execution.

## 10. Multi-process and future PostgreSQL support

- One FastAPI process creates one manager, but several processes on the same host
  may share the local SQLite queue and divide jobs through claims.
- Destination paths must be visible on the same host filesystem.
- Cross-host managers require shared destination storage and a PostgreSQL queue.
- Queue persistence is behind a `QueueStore` protocol so the public manager API
  does not change when PostgreSQL is added.
- A PostgreSQL backend should use `SELECT ... FOR UPDATE SKIP LOCKED`, leases,
  heartbeats, and the same fencing-token checks.
- `LISTEN/NOTIFY` may reduce polling latency, but polling remains the recovery
  fallback.
- Initially, one worker owns an entire model job. Do not distribute chunks of a
  single file across machines.

## 11. Download and retry integrity

- Process one model file at a time initially, using bounded concurrent HTTP
  range requests within that file. This avoids multiplying concurrency and open
  file descriptors across large directories.
- Persist chunk completion only after downloaded bytes are flushed durably.
- Never mark a file complete before size and checksum verification.
- Refresh signed URLs after authentication/expiry failures.
- Reject invalid status, `Content-Range`, `Content-Length`, truncation, excess
  bytes, unsafe URL schemes, and source changes.
- Use restrictive permissions for staging files and queue databases.
- Preserve partial files after ordinary failure or cancellation.
- Process shutdown releases the lease immediately and leaves work queued;
  `cancelled` is reserved for explicit user cancellation.
- Retry temporary network/DNS failures and HTTP `408`, `429`, `500`, `502`,
  `503`, and `504`; honor bounded `Retry-After` and refresh access after
  `401`/`403`.
- Fail immediately on source identity changes, invalid ranges, unsafe paths,
  disk exhaustion, read-only filesystems, and permission failures.
- On final checksum failure, clear completed-chunk claims so retry re-downloads
  all content rather than trusting corrupt bytes.
- Sanitize all durable and operator-facing errors; cap message lengths.

## 12. Destination layout and atomic completion

Download into a job-specific sibling staging directory, for example:

```text
.models-example.<job-id>.partial/
├── config.json.partial
├── model-00001-of-00004.safetensors.partial
└── per-file resume state
```

After every payload file verifies:

1. When checksum or signature verification is enabled, write `SHA256SUMS` using
   the conventional `sha256sum` text format. Never fabricate it when checksum
   verification is explicitly disabled and the server has no manifest.
2. Write `.source.json` containing only validated durable provenance.
3. Flush files and directory metadata as supported by the platform.
4. Atomically rename the staging directory to the final model directory.

The final directory must preserve the upstream model layout; model files must
not be wrapped in a `data/` directory.

## 13. Integrity and provenance files

BagIt (RFC 8493) is a recognized archival-transfer convention, but strict BagIt
would place payload under `data/`, which can break tools expecting a normal
Hugging Face-style model directory. Therefore the downloader uses:

```text
model-directory/
├── original model files...
├── SHA256SUMS
└── .source.json
```

`SHA256SUMS` is the authoritative payload inventory and integrity record. It
uses the conventional `sha256sum` format, UTF-8 encoding, LF endings, entries
sorted by normalized relative path, two spaces between lowercase digest and
path, and one final newline. It includes `.source.json` and every payload file,
and excludes `SHA256SUMS` and `SHA256SUMS.sigstore.json`. Paths must be relative,
normalized, and unable to escape the model directory. By default, the downloader
computes a local SHA-256 for every listed file even when S3 provides no checksum.

`.source.json` contains only durable source provenance. Version 2 includes a
source-file inventory with optional, explicitly non-authoritative
`upstream_sha256` values; `SHA256SUMS` remains the authoritative integrity
record. Its normative schemas and semantics are documented in
[`source_file.md`](source_file.md),
[`source-v1.schema.json`](../src/opai_models/schemas/source-v1.schema.json), and
[`source-v2.schema.json`](../src/opai_models/schemas/source-v2.schema.json). It
contains no URLs, credentials, download progress, or storage-specific resume
identity.

The directory is BagIt-inspired and can be explicitly converted to or from a
BagIt package, but it does not claim BagIt or OCI conformance.

## 14. FastAPI integration

Use FastAPI's current lifespan state pattern, not module-level globals or
`@app.on_event` startup handlers:

```python
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict

from fastapi import Depends, FastAPI, Request
from opai_models import AsyncModelClient, DownloadManager


class AppState(TypedDict):
    downloads: DownloadManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[AppState]:
    async def license_provider() -> str:
        return os.environ["OPAI_LICENSE_KEY"]

    client = AsyncModelClient(
        api_url="https://license.api.onprem.ai",
        license_provider=license_provider,
        sigstore_identity=os.environ["OPAI_SIGSTORE_IDENTITY"],
        sigstore_issuer=os.environ["OPAI_SIGSTORE_ISSUER"],
    )
    manager = DownloadManager(
        database_path=Path("/var/lib/app/model-downloads.sqlite"),
        download_directory=Path("/var/lib/app/models"),
        client=client,
    )
    await manager.start()
    try:
        yield {"downloads": manager}
    finally:
        await manager.close()


app = FastAPI(lifespan=lifespan)


def get_download_manager(request: Request) -> DownloadManager:
    return request.state.downloads


DownloadManagerDep = Annotated[DownloadManager, Depends(get_download_manager)]
```

Typical thin endpoints call `enqueue`, `get`, `list`, `cancel`, and `retry`.
Polling `GET /downloads/{job_id}` reads durable queue state, so status survives
process restarts. FastAPI dependency caching is request-scoped; the lifespan
owns the process-scoped resource.

## 15. RPC and events

- Do not implement an RPC daemon initially.
- Keep queue and manager APIs independent of FastAPI so a future local HTTP/SSE
  or JSON-RPC adapter does not require redesign.
- Preserve a subscription/event boundary for future progress streaming.
- If RPC is added, bind to a Unix-domain socket by default. Remote TCP requires
  a separate TLS and authorization design.

## 16. Security

- Never log or persist licenses, signed URLs, bearer tokens, or authorization
  headers.
- Never include secret-bearing HTTP or validation exceptions verbatim in API or
  CLI output.
- Reject multi-segment model directory names, traversal segments, control
  characters, and unsafe model-relative or destination paths.
- Require HTTPS for production API and signed URLs; permit HTTP only for
  loopback tests.
- Keep partial files, state, SQLite databases, and metadata non-world-readable.
- `SHA256SUMS` must use paths safely and unambiguously; reject newline or
  traversal characters.
- Queue errors expose stable codes plus sanitized messages only.

## 17. Testing and release gates

Required automated coverage includes:

- model-directory recursion and pagination;
- immutable `SHA256SUMS` ordering and snapshot hash;
- multi-file total size and file count;
- ranged transfer response validation;
- cancellation and process-kill resume;
- URL expiry and refresh;
- changed source rejection;
- per-file and complete-model checksum verification;
- atomic directory publication;
- concurrent SQLite claim exclusion;
- lease expiry/reclaim and fencing stale workers;
- heartbeat renewal;
- queued and active cancellation;
- transient retry scheduling, no-progress timeout, and error history;
- restart recovery;
- destination conflict/traversal prevention;
- absence of credentials and signed URLs in logs, databases, and metadata files;
- CLI and async API behavior;
- FastAPI lifespan integration example;
- clean wheel and source-distribution installation on supported Python versions;
- a real read-only License Server listing test before publication.

The project requires Ruff-clean code, at least 90% branch coverage, a frozen lock
file, and successful wheel/sdist builds. Publication remains explicitly out of
scope until separately approved.
