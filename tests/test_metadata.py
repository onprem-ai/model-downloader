import hashlib
from pathlib import Path

import pytest

from opai_models.client import ModelDownloadError
from opai_models.metadata import (
    SourceDocument,
    parse_sha256sums,
    parse_source,
    read_source,
    render_sha256sums,
    snapshot_digest,
    write_source,
)


def source() -> SourceDocument:
    return SourceDocument.from_dict(
        {
            "schema_version": 1,
            "source": {
                "provider": "huggingface",
                "repository": "owner/model",
                "revision": "a" * 40,
                "subdirectory": None,
            },
            "acquisition": {
                "acquired_at": "2026-08-24T09:00:00Z",
                "tool": {"name": "importer", "version": "1"},
            },
            "upstream_metadata": {
                "licenses": ["apache-2.0"],
                "languages": ["en"],
                "tags": ["safe"],
            },
        }
    )


def test_source_round_trip_is_canonical_and_private(tmp_path: Path) -> None:
    path = tmp_path / ".source.json"
    write_source(path, source())
    assert read_source(path) == source()
    assert path.read_bytes().endswith(b"\n")
    assert path.stat().st_mode & 0o777 == 0o600


def test_source_v2_preserves_upstream_file_checksums() -> None:
    value = source().to_dict()
    value["schema_version"] = 2
    value["files"] = [
        {
            "path": "config.json",
            "size": 123,
        },
        {
            "path": "transformer/model.safetensors",
            "size": 456,
            "upstream_sha256": "a" * 64,
        },
    ]

    document = SourceDocument.from_dict(value, require_revision=True)

    assert document.schema_version == 2
    assert document.to_dict()["files"] == value["files"]


def test_source_v2_rejects_noncanonical_file_inventory() -> None:
    value = source().to_dict()
    value["schema_version"] = 2
    invalid_inventories = (
        [],
        [{"path": "../model", "size": 1}],
        [{"path": "model", "size": 0}],
        [{"path": "model", "size": 1, "upstream_sha256": "bad"}],
        [{"path": "b", "size": 1}, {"path": "a", "size": 1}],
        [{"path": "a", "size": 1}, {"path": "a", "size": 1}],
        [{"path": "large", "size": 10_000_000_000_001}],
    )
    for files in invalid_inventories:
        value["files"] = files
        with pytest.raises(ModelDownloadError, match="invalid .source.json"):
            SourceDocument.from_dict(value)


def test_source_v1_rejects_v2_file_inventory() -> None:
    value = source().to_dict()
    value["files"] = [{"path": "model", "size": 1}]
    with pytest.raises(ModelDownloadError, match="invalid .source.json"):
        SourceDocument.from_dict(value)


def test_source_rejects_unknown_operational_and_invalid_fields() -> None:
    base = source().to_dict()
    for changed in (
        {**base, "url": "https://secret"},
        {**base, "source": {**base["source"], "revision": "main"}},
        {**base, "source": {**base["source"], "repository": "bad"}},
        {**base, "source": {**base["source"], "subdirectory": "../bad"}},
        {**base, "acquisition": {"acquired_at": "yesterday"}},
    ):
        with pytest.raises(ModelDownloadError):
            SourceDocument.from_dict(changed)


def test_unknown_historical_revision_is_explicitly_supported() -> None:
    value = source().to_dict()
    value["source"]["revision"] = None
    assert SourceDocument.from_dict(value).source.revision is None
    with pytest.raises(ModelDownloadError, match="immutable revision"):
        SourceDocument.from_dict(value, require_revision=True)


def test_source_omits_absent_tool_version_and_requires_canonical_utc() -> None:
    value = source().to_dict()
    value["acquisition"]["tool"].pop("version")
    parsed = SourceDocument.from_dict(value)
    assert "version" not in parsed.to_dict()["acquisition"]["tool"]

    value["acquisition"]["acquired_at"] = "2026-08-24T11:00:00+02:00"
    with pytest.raises(ModelDownloadError, match="invalid .source.json"):
        SourceDocument.from_dict(value)

    value = source().to_dict()
    value["schema_version"] = True
    with pytest.raises(ModelDownloadError, match="invalid .source.json"):
        SourceDocument.from_dict(value)


def test_sha256sums_round_trip_order_and_snapshot_digest() -> None:
    checksums = {"z file": "a" * 64, "a/config.json": "B" * 64}
    encoded = render_sha256sums(checksums)
    assert encoded == ("b" * 64 + "  a/config.json\n" + "a" * 64 + "  z file\n").encode()
    assert parse_sha256sums(encoded) == {
        "a/config.json": "b" * 64,
        "z file": "a" * 64,
    }
    assert snapshot_digest(encoded) == "sha256:" + hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize(
    "data",
    [
        b"bad  file\n",
        ("a" * 64 + " file\n").encode(),
        ("a" * 64 + "  ../file\n").encode(),
        ("a" * 64 + "  /file\n").encode(),
        ("a" * 64 + "  file\r\n").encode(),
        ("a" * 64 + "  same\n" + "b" * 64 + "  same\n").encode(),
        ("a" * 64 + "  SHA256SUMS\n").encode(),
    ],
)
def test_sha256sums_rejects_noncanonical_or_unsafe_input(data: bytes) -> None:
    with pytest.raises(ModelDownloadError):
        parse_sha256sums(data)


def test_source_rejects_duplicate_json_keys_and_nonfinite_values() -> None:
    for data in (
        b'{"schema_version":1,"schema_version":1,"source":{}}',
        b'{"schema_version":NaN,"source":{}}',
    ):
        with pytest.raises(ModelDownloadError, match="invalid .source.json"):
            parse_source(data)


def test_source_invalid_json_does_not_echo_untrusted_contents(tmp_path: Path) -> None:
    path = tmp_path / ".source.json"
    path.write_text('{"secret":"value"}')
    with pytest.raises(ModelDownloadError, match="invalid .source.json") as caught:
        read_source(path)
    assert "value" not in str(caught.value)


def test_source_read_error_preserves_filesystem_detail(tmp_path: Path) -> None:
    path = tmp_path / "missing" / ".source.json"

    with pytest.raises(ModelDownloadError) as caught:
        read_source(path)

    message = str(caught.value)
    assert "cannot read .source.json" in message
    assert "No such file or directory" in message
