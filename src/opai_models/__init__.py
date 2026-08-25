"""OnPrem AI model downloader public API."""

from opai_models.async_client import AsyncModelClient
from opai_models.client import LicenseClient, LicenseKeyProvider, ModelAccess, ModelDownloadError
from opai_models.download import DownloadCancelled
from opai_models.manager import (
    DownloadError,
    DownloadFile,
    DownloadJob,
    DownloadManager,
    JobConflictError,
    JobNotFoundError,
)
from opai_models.metadata import SourceDocument
from opai_models.signatures import SigstoreIdentity, verify_sigstore_bundle
from opai_models.snapshot import ModelFile, ModelSnapshot
from opai_models.sync import SyncResult

__version__ = "0.1.0"
__all__ = [
    "AsyncModelClient",
    "DownloadCancelled",
    "DownloadError",
    "DownloadFile",
    "DownloadJob",
    "DownloadManager",
    "JobConflictError",
    "JobNotFoundError",
    "LicenseClient",
    "LicenseKeyProvider",
    "ModelAccess",
    "ModelDownloadError",
    "ModelFile",
    "ModelSnapshot",
    "SigstoreIdentity",
    "SourceDocument",
    "SyncResult",
    "verify_sigstore_bundle",
]
