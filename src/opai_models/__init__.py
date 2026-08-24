"""OnPrem AI model downloader public API."""

from opai_models.async_client import AsyncModelClient
from opai_models.client import LicenseClient, ModelAccess, ModelDownloadError
from opai_models.download import DownloadCancelled
from opai_models.manager import (
    DownloadFile,
    DownloadJob,
    DownloadManager,
    JobConflictError,
    JobNotFoundError,
)
from opai_models.metadata import SourceDocument
from opai_models.signatures import SigstoreIdentity, verify_sigstore_bundle
from opai_models.snapshot import ModelFile, ModelSnapshot

__version__ = "0.1.0"
__all__ = [
    "AsyncModelClient",
    "DownloadCancelled",
    "DownloadFile",
    "DownloadJob",
    "DownloadManager",
    "JobConflictError",
    "JobNotFoundError",
    "LicenseClient",
    "ModelAccess",
    "ModelDownloadError",
    "ModelFile",
    "ModelSnapshot",
    "SigstoreIdentity",
    "SourceDocument",
    "verify_sigstore_bundle",
]
