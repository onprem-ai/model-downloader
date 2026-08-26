"""Strict model provenance and portable SHA-256 inventory helpers."""

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from opai_models.client import ModelDownloadError
from opai_models.errors import sanitize_error_detail

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_REPOSITORY = re.compile(r"^[^/\s\x00-\x1f\x7f]+/[^/\s\x00-\x1f\x7f]+$")
_METADATA_NAMES = frozenset({"SHA256SUMS", "SHA256SUMS.sigstore.json"})
_SOURCE_RESERVED_NAMES = frozenset({".source.json", *_METADATA_NAMES})
_MAX_SOURCE_FILES = 10_000
_MAX_SOURCE_BYTES = 10_000_000_000_000


def safe_relative_path(value: str) -> str:
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ModelDownloadError("unsafe relative path")
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts) or str(path) != value:
        raise ModelDownloadError("unsafe relative path")
    if path.name in _METADATA_NAMES:
        raise ModelDownloadError("package metadata cannot be a payload file")
    return value


@dataclass(frozen=True)
class Source:
    provider: str
    repository: str
    revision: str | None
    subdirectory: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    version: str | None = None


@dataclass(frozen=True)
class Acquisition:
    acquired_at: str
    tool: Tool | None = None


@dataclass(frozen=True)
class UpstreamMetadata:
    licenses: tuple[str, ...] = ()
    library_name: str | None = None
    pipeline_tag: str | None = None
    languages: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int
    upstream_sha256: str | None = None


@dataclass(frozen=True)
class SourceDocument:
    schema_version: int
    source: Source
    acquisition: Acquisition | None = None
    upstream_metadata: UpstreamMetadata | None = None
    files: tuple[SourceFile, ...] = ()

    @classmethod
    def from_dict(cls, value: Any, *, require_revision: bool = False) -> "SourceDocument":
        try:
            if not isinstance(value, dict) or type(value.get("schema_version")) is not int:
                raise ValueError
            schema_version = value["schema_version"]
            allowed = {"schema_version", "source", "acquisition", "upstream_metadata"}
            required = {"schema_version", "source"}
            if schema_version == 2:
                allowed.add("files")
                required.add("files")
            elif schema_version != 1:
                raise ValueError
            _keys(value, allowed, required)
            raw_source = value["source"]
            _keys(
                raw_source,
                {"provider", "repository", "revision", "subdirectory"},
                {"provider", "repository", "revision"},
            )
            revision = raw_source["revision"]
            if raw_source["provider"] != "huggingface" or not _REPOSITORY.fullmatch(
                raw_source["repository"]
            ):
                raise ValueError
            if revision is not None and (
                not isinstance(revision, str) or not _REVISION.fullmatch(revision)
            ):
                raise ValueError
            if require_revision and revision is None:
                raise ModelDownloadError("new models require an immutable revision")
            subdirectory = raw_source.get("subdirectory")
            if subdirectory is not None:
                safe_relative_path(subdirectory)
            source = Source("huggingface", raw_source["repository"], revision, subdirectory)
            acquisition = _parse_acquisition(value.get("acquisition"))
            metadata = _parse_metadata(value.get("upstream_metadata"))
            try:
                files = _parse_source_files(value.get("files")) if schema_version == 2 else ()
            except ModelDownloadError:
                raise ValueError from None
            return cls(schema_version, source, acquisition, metadata, files)
        except ModelDownloadError:
            raise
        except (KeyError, TypeError, ValueError):
            raise ModelDownloadError("invalid .source.json") from None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source": asdict(self.source),
        }
        if self.acquisition is not None:
            acquisition: dict[str, Any] = {"acquired_at": self.acquisition.acquired_at}
            if self.acquisition.tool is not None:
                acquisition["tool"] = {
                    key: value
                    for key, value in asdict(self.acquisition.tool).items()
                    if value is not None
                }
            result["acquisition"] = acquisition
        if self.upstream_metadata is not None:
            metadata = asdict(self.upstream_metadata)
            metadata["licenses"] = list(metadata["licenses"])
            metadata["languages"] = list(metadata["languages"])
            metadata["tags"] = list(metadata["tags"])
            result["upstream_metadata"] = {
                k: v for k, v in metadata.items() if v not in (None, [], ())
            }
        if self.schema_version == 2:
            result["files"] = [
                {key: value for key, value in asdict(item).items() if value is not None}
                for item in self.files
            ]
        return result


def _parse_source_files(value: Any) -> tuple[SourceFile, ...]:
    if not isinstance(value, list) or not value or len(value) > _MAX_SOURCE_FILES:
        raise ValueError
    files: list[SourceFile] = []
    previous_path: str | None = None
    total_size = 0
    for item in value:
        _keys(item, {"path", "size", "upstream_sha256"}, {"path", "size"})
        path = safe_relative_path(item["path"])
        if path in _SOURCE_RESERVED_NAMES:
            raise ValueError
        if previous_path is not None and path <= previous_path:
            raise ValueError
        size = item["size"]
        if type(size) is not int or size <= 0:
            raise ValueError
        total_size += size
        if total_size > _MAX_SOURCE_BYTES:
            raise ValueError
        upstream_sha256 = item.get("upstream_sha256")
        if upstream_sha256 is not None and (
            not isinstance(upstream_sha256, str) or not _SHA256.fullmatch(upstream_sha256)
        ):
            raise ValueError
        files.append(SourceFile(path, size, upstream_sha256))
        previous_path = path
    return tuple(files)


def _keys(value: Any, allowed: set[str], required: set[str]) -> None:
    if not isinstance(value, dict) or not required <= value.keys() or not value.keys() <= allowed:
        raise ValueError


def _text(value: Any, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError
    return value


def _parse_acquisition(value: Any) -> Acquisition | None:
    if value is None:
        return None
    _keys(value, {"acquired_at", "tool"}, {"acquired_at"})
    timestamp = _text(value["acquired_at"], 64)
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None or not timestamp.endswith("Z"):
        raise ValueError
    tool_value = value.get("tool")
    tool = None
    if tool_value is not None:
        _keys(tool_value, {"name", "version"}, {"name"})
        tool = Tool(
            _text(tool_value["name"], 128),
            _text(tool_value["version"], 128) if "version" in tool_value else None,
        )
    return Acquisition(timestamp, tool)


def _parse_metadata(value: Any) -> UpstreamMetadata | None:
    if value is None:
        return None
    allowed = {"licenses", "library_name", "pipeline_tag", "languages", "tags"}
    _keys(value, allowed, set())

    def values(name: str, maximum: int) -> tuple[str, ...]:
        raw = value.get(name, [])
        if not isinstance(raw, list) or len(raw) != len(set(raw)):
            raise ValueError
        return tuple(_text(item, maximum) for item in raw)

    return UpstreamMetadata(
        values("licenses", 128),
        _text(value["library_name"], 128) if "library_name" in value else None,
        _text(value["pipeline_tag"], 128) if "pipeline_tag" in value else None,
        values("languages", 64),
        values("tags", 256),
    )


def parse_source(data: bytes, *, require_revision: bool = False) -> SourceDocument:
    """Parse provenance while rejecting duplicate keys and non-standard constants."""

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate key")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise ModelDownloadError("invalid .source.json") from None
    return SourceDocument.from_dict(value, require_revision=require_revision)


def read_source(path: Path, *, require_revision: bool = False) -> SourceDocument:
    try:
        data = path.read_bytes()
    except OSError as exc:
        detail = sanitize_error_detail(exc)
        raise ModelDownloadError(f"cannot read .source.json: {detail}") from None
    return parse_source(data, require_revision=require_revision)


def write_source(path: Path, document: SourceDocument) -> None:
    data = (json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n").encode()
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def render_sha256sums(checksums: dict[str, str]) -> bytes:
    entries: list[str] = []
    for path in sorted(checksums):
        safe_relative_path(path)
        digest = checksums[path].lower()
        if not _SHA256.fullmatch(digest):
            raise ModelDownloadError("invalid SHA-256 digest")
        entries.append(f"{digest}  {path}\n")
    if not entries:
        raise ModelDownloadError("checksum inventory must not be empty")
    return "".join(entries).encode("utf-8")


def parse_sha256sums(data: bytes) -> dict[str, str]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise ModelDownloadError("invalid SHA256SUMS") from None
    if not text.endswith("\n") or "\r" in text:
        raise ModelDownloadError("invalid SHA256SUMS")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise ModelDownloadError("invalid SHA256SUMS")
        digest, path = line[:64], line[66:]
        if not _SHA256.fullmatch(digest):
            raise ModelDownloadError("invalid SHA256SUMS")
        safe_relative_path(path)
        if path in result:
            raise ModelDownloadError("duplicate path in SHA256SUMS")
        result[path] = digest
    if render_sha256sums(result) != data:
        raise ModelDownloadError("SHA256SUMS is not canonical")
    return result


def snapshot_digest(data: bytes) -> str:
    parse_sha256sums(data)
    return "sha256:" + hashlib.sha256(data).hexdigest()
