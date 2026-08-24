"""Immutable model-directory inventory creation."""

import base64
import hashlib
import re
from dataclasses import dataclass

from opai_models.client import LicenseClient, ModelDownloadError
from opai_models.metadata import (
    SourceDocument,
    parse_sha256sums,
    parse_source,
    render_sha256sums,
    snapshot_digest,
)
from opai_models.signatures import SigstoreIdentity, verify_sigstore_bundle

_METADATA_NAMES = frozenset({"SHA256SUMS", "SHA256SUMS.sigstore.json"})


@dataclass(frozen=True)
class ModelFile:
    object_path: str
    relative_path: str
    size: int
    source_id: str
    sha256: str
    etag: str | None = None
    version_id: str | None = None


@dataclass(frozen=True)
class ModelSnapshot:
    model_id: str
    files: tuple[ModelFile, ...]
    file_count: int
    total_bytes: int
    sha256sums: str
    snapshot_sha256: str
    source: SourceDocument | None = None

    @classmethod
    def create(
        cls,
        model_id: str,
        files: list[ModelFile],
        source: SourceDocument | None = None,
    ) -> "ModelSnapshot":
        ordered = tuple(sorted(files, key=lambda item: item.relative_path))
        if not ordered:
            raise ModelDownloadError("model directory is empty")
        encoded = render_sha256sums({item.relative_path: item.sha256 for item in ordered})
        return cls(
            model_id,
            ordered,
            len(ordered),
            sum(item.size for item in ordered),
            encoded.decode(),
            snapshot_digest(encoded),
            source,
        )


def _sha256(value: str | None) -> str | None:
    if not value:
        return None
    if re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return value.lower()
    try:
        decoded = base64.b64decode(value, validate=True)
    except ValueError:
        return None
    return decoded.hex() if len(decoded) == 32 else None


def snapshot_model(
    client: LicenseClient,
    model_id: str,
    *,
    verify_checksums: bool = True,
    verify_signatures: bool = True,
    trusted_identity: SigstoreIdentity | None = None,
    sigstore_offline: bool = False,
) -> ModelSnapshot:
    model = LicenseClient._model_id(model_id)
    listings: list[dict[str, object]] = []
    pending = [""]
    visited: set[str] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        listing = client.list_all(model, current)
        listings.append(listing)
        for child in listing.get("prefixes") or []:
            child_prefix = str(child)
            if current and not child_prefix.startswith(current):
                raise ModelDownloadError(
                    "model listing returned a prefix outside the requested prefix"
                )
            LicenseClient._relative_path(child_prefix, prefix=True)
            pending.append(child_prefix)

    listed: dict[str, int] = {}
    metadata: set[str] = set()
    for listing in listings:
        for item in listing.get("objects") or []:
            try:
                object_path = str(item["key"])
                listed_size = int(item["size"])
            except (KeyError, TypeError, ValueError):
                raise ModelDownloadError(
                    "License Server returned an invalid model listing"
                ) from None
            relative = LicenseClient._relative_path(object_path)
            listing_prefix = str(listing.get("prefix") or "")
            if listing_prefix and not relative.startswith(listing_prefix):
                raise ModelDownloadError(
                    "model listing returned an object outside the requested prefix"
                )
            if relative in _METADATA_NAMES:
                metadata.add(relative)
            elif relative:
                if relative in listed:
                    raise ModelDownloadError("model listing contains a duplicate object")
                listed[relative] = listed_size

    if "SHA256SUMS" not in metadata or ".source.json" not in listed:
        raise ModelDownloadError("model requires SHA256SUMS and .source.json")
    checksum_bytes = client.read_small(model, "SHA256SUMS")
    checksums = parse_sha256sums(checksum_bytes)
    if set(checksums) != set(listed):
        raise ModelDownloadError("SHA256SUMS inventory does not match model objects")
    source_bytes = client.read_small(model, ".source.json")
    source = parse_source(source_bytes)
    if verify_checksums and hashlib.sha256(source_bytes).hexdigest() != checksums[".source.json"]:
        raise ModelDownloadError(".source.json checksum verification failed")
    if verify_signatures:
        if trusted_identity is None:
            raise ModelDownloadError("Sigstore trusted identity and issuer are required")
        if "SHA256SUMS.sigstore.json" not in metadata:
            raise ModelDownloadError("model signature is required")
        bundle = client.read_small(model, "SHA256SUMS.sigstore.json", maximum=16 * 1024 * 1024)
        verify_sigstore_bundle(
            checksum_bytes,
            bundle,
            trusted_identity,
            offline=sigstore_offline,
        )

    files: list[ModelFile] = []
    for relative, expected_sha256 in checksums.items():
        object_path = relative
        access = client.access(model, object_path)
        if access.path != object_path:
            raise ModelDownloadError("model access path does not match listing")
        if access.size != listed[relative]:
            raise ModelDownloadError("model file size changed while snapshotting")
        provider_sha256 = _sha256(access.checksums.get("sha256"))
        if verify_checksums and provider_sha256 is not None and provider_sha256 != expected_sha256:
            raise ModelDownloadError("model file checksum changed while snapshotting")
        files.append(
            ModelFile(
                object_path,
                relative,
                access.size,
                access.source_id,
                expected_sha256,
                access.etag,
                access.version_id,
            )
        )
    return ModelSnapshot.create(model, files, source)
