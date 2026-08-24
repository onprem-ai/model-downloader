import base64
import hashlib

from opai_models.download import _expected_sha256, _ranges


def test_ranges_cover_object_exactly() -> None:
    assert _ranges(10, 4) == [(0, 0, 3), (1, 4, 7), (2, 8, 9)]


def test_sha256_accepts_hex_and_base64() -> None:
    digest = hashlib.sha256(b"test").digest()
    assert _expected_sha256({"sha256": digest.hex()}) == digest
    assert _expected_sha256({"sha256": base64.b64encode(digest).decode()}) == digest
    assert _expected_sha256({"sha256": "bad"}) is None
