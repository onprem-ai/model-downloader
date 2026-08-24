# opai-models

`opai-models` is an asynchronous Python library and command-line client for
securely downloading complete model-weight directories from OnPrem AI storage.
Downloads use short-lived URLs issued by the License Server; license keys and
signed URLs are never persisted.

## Installation

Python 3.11 or newer is required.

Once a package release is published:

```bash
pip install opai-models
```

Until then, install from a checked-out repository:

```bash
git clone https://github.com/onprem-ai/model-downloader.git
cd model-downloader
uv sync --frozen
```

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
  --retries 3
```

The argument is the model ID, not an S3 path. The downloader never needs a
bucket name, S3 endpoint, storage prefix, or permanent S3 credential. The
License Server maps the model ID to storage and returns short-lived download
URLs.

The default API is `https://license.api.onprem.ai`. Interrupted downloads resume
from the local SQLite queue and chunk state. Keep that database on local storage,
not NFS.

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

client = AsyncModelClient(
    "https://license.api.onprem.ai",
    license_provider=lambda: os.environ["OPAI_LICENSE_KEY"],
    sigstore_identity=os.environ["OPAI_SIGSTORE_IDENTITY"],
    sigstore_issuer=os.environ["OPAI_SIGSTORE_ISSUER"],
)
manager = DownloadManager(
    database_path=Path("/var/lib/app/model-downloads.sqlite"),
    download_directory=Path("/var/lib/app/models"),
    client=client,
)

async with manager:
    job = await manager.enqueue("example")
    completed = await manager.wait(job.id)
```

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
