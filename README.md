# opai-models

`opai-models` is an asynchronous Python library and command-line client for
securely downloading complete model-weight directories from OnPrem AI storage.
Downloads use short-lived URLs issued by the License Server; license keys and
signed URLs are never persisted.

## Installation

Python 3.11 or newer is required.

The recommended installation uses the committed `uv.lock` file so the complete
runtime dependency graph is reproducible:

```bash
git clone https://github.com/onprem-ai/model-downloader.git
cd model-downloader
uv sync --frozen
uv run --frozen --no-sync opai-models --help
```

Once a package release is published, `pip install opai-models` will be available
as a convenience alternative, but it will not reproduce the complete dependency
set from this repository's `uv.lock` file.

## Authentication

Provide the OnPrem AI license key through the environment:

```bash
export OPAI_LICENSE_KEY='<license-key>'
```

If the variable is absent in an interactive terminal, the CLI prompts without
echoing the key. Use `--license-env NAME` to select another variable.

Signature verification is enabled by default and requires the exact trusted
Sigstore certificate identity and OIDC issuer:

```bash
export OPAI_SIGSTORE_IDENTITY='<trusted-certificate-identity>'
export OPAI_SIGSTORE_ISSUER='<trusted-oidc-issuer>'
```

## CLI

List available models:

```bash
opai-models list
```

Download a complete model directory:

```bash
opai-models pull example
```

Choose storage paths and transfer settings:

```bash
opai-models pull example example \
  --download-directory /srv/models \
  --database /var/lib/opai-models/downloads.sqlite \
  --chunk-size 67108864 \
  --workers 4 \
  --request-retries 8 \
  --integrity-retries 2 \
  --initial-backoff 0.5 \
  --max-backoff 60 \
  --no-progress-timeout 3600 \
  --overall-timeout 0
```

Synchronize an existing completed directory without relying on SQLite state:

```bash
opai-models sync example /srv/models/example
```

If the local and remote `SHA256SUMS` files are identical, existing files with
matching sizes are reused without rehashing. If manifests differ, only files
with matching manifest entries are reused directly; other existing files are
hashed and downloaded only when missing or invalid. Use `--rehash` to hash every
local model file, and `--delete` to omit files that are no longer in the remote
manifest. Synchronization builds a hidden sibling replacement directory and
uses a rollback-safe replacement only after all required files are ready.

The argument is the model directory name, not an S3 path. The downloader never
needs a bucket name, S3 endpoint, storage prefix, or permanent S3 credential.
The License Server maps the model directory name to storage and returns
short-lived download URLs.

The default API is `https://license.api.onprem.ai`. Interrupted downloads resume
from durable per-file, per-chunk state. Transient failures enter `retry_wait` and
resume automatically with exponential full-jitter backoff. A successful chunk
resets the consecutive-failure count; the job fails only after the configured
no-progress timeout or a permanent error. Detailed sanitized errors are retained
against the affected job, file, and chunk. Keep the SQLite database on local
storage, not NFS. If the database schema is incompatible, delete the database
and let the downloader create a new one.

### Verification controls

Checksum and signature verification are both enabled by default. They can be
disabled independently when explicitly required:

```bash
# Accept an unsigned manifest, but still verify every downloaded file.
opai-models pull example --skip-signature-verification

# Verify the manifest signature, but skip hashing downloaded payload bytes.
opai-models pull example --skip-checksum-verification

# Disable both protections. This is not recommended.
opai-models pull example \
  --skip-checksum-verification \
  --skip-signature-verification
```

`--sigstore-offline` uses cached Sigstore trust roots without refreshing them.
Offline deployments are responsible for distributing current trusted roots.

## Python API

```python
import os
from pathlib import Path

from opai_models import AsyncModelClient, DownloadManager

async def license_provider() -> str:
    return os.environ["OPAI_LICENSE_KEY"]


client = AsyncModelClient(
    "https://license.api.onprem.ai",
    license_provider=license_provider,
    sigstore_identity=os.environ["OPAI_SIGSTORE_IDENTITY"],
    sigstore_issuer=os.environ["OPAI_SIGSTORE_ISSUER"],
)
manager = DownloadManager(
    database_path=Path("/var/lib/app/model-downloads.sqlite"),
    download_directory=Path("/var/lib/app/models"),
    client=client,
)

try:
    async with manager:
        job = await manager.enqueue("example")
        completed = await manager.wait(job.id)
finally:
    await client.aclose()
```

The client uses a reusable native-async HTTPX connection pool. The async license
provider is awaited for each authenticated API request. Credentials remain in
memory and are not stored in the downloader database or model directory.

Progress callbacks are async and are awaited on the event loop before the
operation continues. A callback that needs blocking I/O must explicitly offload it.

For a web service, create exactly one manager per application process in the
FastAPI lifespan. See the complete commented example:

[`examples/fastapi_singleton.py`](examples/fastapi_singleton.py)

## Model directory contract

A model directory retains its normal runtime layout and adds:

```text
model/
├── payload files and subdirectories...
├── .source.json
├── SHA256SUMS
└── SHA256SUMS.sigstore.json
```

`SHA256SUMS` is canonical and includes `.source.json` plus every payload file. It
excludes itself and `SHA256SUMS.sigstore.json`. The signature bundle authenticates
the exact `SHA256SUMS` bytes.

The downloader:

- recursively inventories the model directory;
- validates `.source.json`;
- verifies the Sigstore bundle against the configured identity and issuer;
- downloads with bounded concurrent ranged requests;
- verifies each downloaded file using SHA-256;
- resumes interrupted transfers from SQLite;
- atomically publishes the completed directory;
- rejects traversal and symbolic-link attacks.

See [`docs/source_file.md`](docs/source_file.md) for provenance details and
[`docs/requirements.md`](docs/requirements.md) for the complete design and
security requirements.

## Development

```bash
uv lock --check
uv sync --frozen
uv run ruff check .
uv run pytest -q --cov=opai_models --cov-branch --cov-fail-under=90
uv build
```

Licensed under the Apache License 2.0.
