import hashlib
import json
from unittest.mock import MagicMock

import pytest

from opai_models.client import ModelAccess, ModelDownloadError, _AsyncLicenseTransport
from opai_models.metadata import SourceDocument, render_sha256sums
from opai_models.snapshot import ModelFile, ModelSnapshot, _sha256, snapshot_model


def access(path: str, size: int, digest: str | None = None) -> ModelAccess:
    checksums = {"sha256": digest} if digest else {}
    return ModelAccess(
        path,
        "https://s3.example/signed",
        size,
        path + ":id",
        "later",
        checksums,
        {},
        "etag",
        "version",
    )


def source_bytes(revision: str | None = "a" * 40) -> bytes:
    return (
        __import__("json")
        .dumps(
            {
                "schema_version": 1,
                "source": {
                    "provider": "huggingface",
                    "repository": "owner/model",
                    "revision": revision,
                },
            }
        )
        .encode()
    )


def listing() -> list[dict[str, object]]:
    source = source_bytes()
    hashes = {
        ".source.json": hashlib.sha256(source).hexdigest(),
        "a.json": hashlib.sha256(b"a").hexdigest(),
        "sub/nested.bin": hashlib.sha256(b"zz").hexdigest(),
        "z.bin": hashlib.sha256(b"zz").hexdigest(),
    }
    sums = render_sha256sums(hashes)
    return [
        {
            "prefix": "",
            "objects": [
                {"key": "z.bin", "size": 2},
                {"key": ".source.json", "size": len(source)},
                {"key": "SHA256SUMS", "size": len(sums)},
                {"key": "a.json", "size": 1},
            ],
            "prefixes": ["sub/"],
        },
        {
            "prefix": "sub/",
            "objects": [{"key": "sub/nested.bin", "size": 2}],
            "prefixes": [],
        },
    ]


@pytest.mark.asyncio
async def test_snapshot_checksum_normalization_and_empty_inventory() -> None:
    digest = "a" * 64
    assert _sha256(digest.upper()) == digest
    assert _sha256(__import__("base64").b64encode(bytes.fromhex(digest)).decode()) == digest
    assert _sha256(None) is None
    assert _sha256("not-a-checksum") is None
    assert _sha256(__import__("base64").b64encode(b"short").decode()) is None
    with pytest.raises(ModelDownloadError, match="empty"):
        ModelSnapshot.create("example", [])


@pytest.mark.asyncio
async def test_snapshot_rejects_empty_remote_model_before_manifest_validation() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    client.list_all.return_value = {
        "prefix": "",
        "objects": [],
        "prefixes": [],
    }

    with pytest.raises(
        ModelDownloadError,
        match="remote model directory does not exist or is empty: example",
    ):
        await snapshot_model(client, "example", verify_signatures=False)

    client.read_small.assert_not_awaited()


@pytest.mark.asyncio
async def test_snapshot_recurses_and_uses_authoritative_metadata() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    pages = listing()
    client.list_all.side_effect = pages
    sums = render_sha256sums(
        {
            ".source.json": hashlib.sha256(source_bytes()).hexdigest(),
            "a.json": hashlib.sha256(b"a").hexdigest(),
            "sub/nested.bin": hashlib.sha256(b"zz").hexdigest(),
            "z.bin": hashlib.sha256(b"zz").hexdigest(),
        }
    )
    client.read_small.side_effect = [sums, source_bytes()]
    sizes = {
        ".source.json": len(source_bytes()),
        "a.json": 1,
        "sub/nested.bin": 2,
        "z.bin": 2,
    }
    client.access.side_effect = lambda model, path: access(path, sizes[path])
    result = await snapshot_model(client, "example", verify_signatures=False)
    assert [item.relative_path for item in result.files] == [
        ".source.json",
        "a.json",
        "sub/nested.bin",
        "z.bin",
    ]
    assert result.file_count == 4
    assert result.total_bytes == 5 + len(source_bytes())
    assert result.sha256sums == sums.decode()
    assert result.source.source.repository == "owner/model"


@pytest.mark.asyncio
async def test_snapshot_requires_trusted_identity_when_signature_exists() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    pages = listing()
    pages[0]["objects"].append({"key": "SHA256SUMS.sigstore.json", "size": 8})
    client.list_all.side_effect = pages
    source = source_bytes()
    sums = render_sha256sums(
        {
            ".source.json": hashlib.sha256(source).hexdigest(),
            "a.json": hashlib.sha256(b"a").hexdigest(),
            "sub/nested.bin": hashlib.sha256(b"zz").hexdigest(),
            "z.bin": hashlib.sha256(b"zz").hexdigest(),
        }
    )
    client.read_small.side_effect = [sums, source]
    with pytest.raises(ModelDownloadError, match="trusted identity"):
        await snapshot_model(client, "example")


@pytest.mark.asyncio
async def test_snapshot_enforces_signature_by_default_and_passes_exact_manifest() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    pages = listing()
    pages[0]["objects"].append({"key": "SHA256SUMS.sigstore.json", "size": 8})
    client.list_all.side_effect = pages
    source = source_bytes()
    sums = render_sha256sums(
        {
            ".source.json": hashlib.sha256(source).hexdigest(),
            "a.json": hashlib.sha256(b"a").hexdigest(),
            "sub/nested.bin": hashlib.sha256(b"zz").hexdigest(),
            "z.bin": hashlib.sha256(b"zz").hexdigest(),
        }
    )
    client.read_small.side_effect = [sums, source, b"bundle"]
    sizes = {
        ".source.json": len(source),
        "a.json": 1,
        "sub/nested.bin": 2,
        "z.bin": 2,
    }
    client.access.side_effect = lambda model, path: access(path, sizes[path])
    from opai_models.signatures import SigstoreIdentity

    identity = SigstoreIdentity("identity", "https://issuer")
    with __import__("unittest.mock").mock.patch(
        "opai_models.snapshot.verify_sigstore_bundle"
    ) as verify:
        await snapshot_model(client, "example", trusted_identity=identity)
    verify.assert_called_once_with(sums, b"bundle", identity, offline=False)

    client.list_all.side_effect = listing()
    client.read_small.side_effect = [sums, source]
    with pytest.raises(ModelDownloadError, match="signature is required"):
        await snapshot_model(client, "example", trusted_identity=identity)


@pytest.mark.asyncio
async def test_checksum_verification_can_be_skipped_independently() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    client.list_all.return_value = {
        "objects": [
            {"key": ".source.json", "size": len(source_bytes())},
            {"key": "file", "size": 1},
            {"key": "SHA256SUMS", "size": 1},
        ],
        "prefixes": [],
    }
    client.read_small.return_value = source_bytes()
    client.access.side_effect = lambda model, path: access(
        path, len(source_bytes()) if path.endswith(".source.json") else 1, "f" * 64
    )
    result = await snapshot_model(
        client,
        "example",
        verify_checksums=False,
        verify_signatures=False,
    )
    assert result.file_count == 2
    assert all(item.sha256 is None for item in result.files)
    assert result.sha256sums is None


@pytest.mark.asyncio
async def test_snapshot_without_verification_uses_authenticated_listing() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    source = source_bytes()
    client.list_all.return_value = {
        "objects": [
            {"key": ".source.json", "size": len(source)},
            {"key": "file", "size": 4},
        ],
        "prefixes": [],
    }
    client.read_small.return_value = source
    client.access.side_effect = lambda model, path: access(
        path, len(source) if path == ".source.json" else 4
    )
    result = await snapshot_model(
        client,
        "example",
        verify_checksums=False,
        verify_signatures=False,
    )
    assert [item.relative_path for item in result.files] == [".source.json", "file"]
    assert all(item.sha256 is None for item in result.files)
    assert result.sha256sums is None
    client.read_small.assert_called_once_with("example", ".source.json")


@pytest.mark.asyncio
async def test_snapshot_rejects_v2_source_inventory_mismatch() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    source = json.dumps(
        {
            "schema_version": 2,
            "source": {
                "provider": "huggingface",
                "repository": "owner/model",
                "revision": "a" * 40,
            },
            "files": [{"path": "other", "size": 4}],
        }
    ).encode()
    checksums = render_sha256sums(
        {
            ".source.json": hashlib.sha256(source).hexdigest(),
            "file": hashlib.sha256(b"data").hexdigest(),
        }
    )
    client.list_all.return_value = {
        "objects": [
            {"key": ".source.json", "size": len(source)},
            {"key": "SHA256SUMS", "size": len(checksums)},
            {"key": "file", "size": 4},
        ],
        "prefixes": [],
    }
    client.read_small.side_effect = [checksums, source]

    with pytest.raises(ModelDownloadError, match="source.json file inventory"):
        await snapshot_model(client, "example", verify_signatures=False)


@pytest.mark.asyncio
async def test_snapshot_requires_metadata_and_exact_inventory() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    client.list_all.return_value = {
        "objects": [{"key": ".source.json", "size": len(source_bytes())}],
        "prefixes": [],
    }
    with pytest.raises(ModelDownloadError, match="SHA256SUMS"):
        await snapshot_model(client, "example", verify_signatures=False)

    client.list_all.return_value = listing()[0] | {"prefixes": []}
    client.read_small.side_effect = [
        render_sha256sums({"only": "a" * 64}),
        source_bytes(),
    ]
    with pytest.raises(ModelDownloadError, match="inventory"):
        await snapshot_model(client, "example", verify_signatures=False)


@pytest.mark.asyncio
async def test_snapshot_rejects_changed_listing_and_provider_checksum() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    client.list_all.return_value = {
        "objects": [
            {"key": "file", "size": 4},
            {"key": "SHA256SUMS", "size": 72},
            {"key": ".source.json", "size": len(source_bytes())},
        ],
        "prefixes": [],
    }
    client.read_small.side_effect = [
        render_sha256sums(
            {
                ".source.json": hashlib.sha256(source_bytes()).hexdigest(),
                "file": "a" * 64,
            }
        ),
        source_bytes(),
    ]
    client.access.side_effect = lambda model, path: access(
        path, len(source_bytes()) if path.endswith(".source.json") else 3
    )
    with pytest.raises(ModelDownloadError, match="size changed"):
        await snapshot_model(client, "example", verify_signatures=False)
    client.read_small.side_effect = [
        render_sha256sums(
            {
                ".source.json": hashlib.sha256(source_bytes()).hexdigest(),
                "file": "a" * 64,
            }
        ),
        source_bytes(),
    ]
    client.access.side_effect = lambda model, path: access(
        path,
        len(source_bytes()) if path.endswith(".source.json") else 4,
        hashlib.sha256(source_bytes()).hexdigest() if path.endswith(".source.json") else "b" * 64,
    )
    with pytest.raises(ModelDownloadError, match="checksum changed"):
        await snapshot_model(client, "example", verify_signatures=False)

    client.read_small.side_effect = [
        render_sha256sums(
            {
                ".source.json": hashlib.sha256(source_bytes()).hexdigest(),
                "file": "a" * 64,
            }
        ),
        source_bytes(),
    ]
    client.access.side_effect = lambda model, path: access(
        path, len(source_bytes()) if path.endswith(".source.json") else 4
    )
    result = await snapshot_model(client, "example", verify_signatures=False)
    assert next(item for item in result.files if item.relative_path == "file").sha256 == "a" * 64


@pytest.mark.asyncio
async def test_snapshot_rejects_root_and_duplicate_objects() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    with pytest.raises(ModelDownloadError, match="model directory name"):
        await snapshot_model(client, "bad/id")
    client.list_all.return_value = {
        "objects": [
            {"key": "file", "size": 1},
            {"key": "file", "size": 1},
        ],
        "prefixes": [],
    }
    with pytest.raises(ModelDownloadError, match="duplicate"):
        await snapshot_model(client, "example", verify_signatures=False)


@pytest.mark.asyncio
async def test_snapshot_rejects_outside_child_prefix_and_object() -> None:
    client = MagicMock(spec=_AsyncLicenseTransport)
    client.list_all.return_value = {"objects": [], "prefixes": ["../other/"]}
    with pytest.raises(ModelDownloadError, match="model-relative path"):
        await snapshot_model(client, "example", verify_signatures=False)
    client.list_all.return_value = {
        "objects": [
            {"key": "../other/file", "size": 1},
            {"key": "SHA256SUMS", "size": 1},
            {"key": ".source.json", "size": 1},
        ],
        "prefixes": [],
    }
    with pytest.raises(ModelDownloadError, match="model-relative path"):
        await snapshot_model(client, "example", verify_signatures=False)


@pytest.mark.asyncio
async def test_snapshot_records_resume_identity_without_urls() -> None:
    item = ModelFile("f", "f", 1, "source", "a" * 64, "etag", "version")
    source = SourceDocument.from_dict(
        {
            "schema_version": 1,
            "source": {
                "provider": "huggingface",
                "repository": "owner/model",
                "revision": None,
            },
        }
    )
    result = ModelSnapshot.create("a", [item], source)
    assert result.file_count == 1
    assert "https://" not in repr(result)
