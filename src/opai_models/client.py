"""License Server client for model metadata and renewable download access."""

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


class ModelDownloadError(RuntimeError):
    """Safe operator-facing model transfer error."""


class TransientModelDownloadError(ModelDownloadError):
    """A temporary network or service failure suitable for automatic retry."""


@dataclass(frozen=True)
class ModelAccess:
    path: str
    url: str
    size: int
    source_id: str
    expires_at: str
    checksums: dict[str, str]
    required_headers: dict[str, str]
    etag: str | None = None
    version_id: str | None = None


class LicenseClient:
    def __init__(self, api_url: str, license_key: str, timeout: float = 30) -> None:
        parsed = urllib.parse.urlparse(api_url)
        loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or (parsed.scheme != "https" and not loopback)
        ):
            raise ModelDownloadError(
                "API URL must be HTTPS (HTTP is allowed only for loopback tests)"
            )
        if not license_key:
            raise ModelDownloadError("license key must not be empty")
        self.api_url = api_url.rstrip("/")
        self.license_key = license_key
        self.timeout = timeout

    @staticmethod
    def _model_id(value: str) -> str:
        if (
            not isinstance(value, str)
            or not value
            or value != value.strip()
            or "/" in value
            or "\\" in value
            or value in {".", ".."}
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ModelDownloadError("invalid model ID")
        return value

    @staticmethod
    def _relative_path(value: str, *, prefix: bool = False) -> str:
        if (
            not isinstance(value, str)
            or not value
            or value.startswith(("/", "\\"))
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ModelDownloadError("invalid model-relative path")
        normalized = value.rstrip("/") if prefix else value
        segments = normalized.split("/")
        if any(segment in {"", ".", ".."} for segment in segments):
            raise ModelDownloadError("invalid model-relative path")
        if not prefix and value.endswith("/"):
            raise ModelDownloadError("model path must identify a file")
        return normalized + "/" if prefix else normalized

    @staticmethod
    def _encoded(path: str) -> str:
        return "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))

    def _json(self, path: str) -> dict[str, Any]:
        request = urllib.request.Request(  # noqa: S310 -- URL validated in __init__
            self.api_url + path,
            headers={
                "Authorization": f"Bearer {self.license_key}",
                "Accept": "application/json",
                "User-Agent": "opai-models/0.1.0",
            },
        )
        try:
            with urllib.request.urlopen(  # noqa: S310 -- URL scheme validated in __init__
                request, timeout=self.timeout
            ) as response:
                body = json.load(response)
        except urllib.error.HTTPError as exc:
            try:
                error_type = (
                    TransientModelDownloadError
                    if exc.code in {408, 429, 500, 502, 503, 504}
                    else ModelDownloadError
                )
                raise error_type(f"License Server returned HTTP {exc.code}") from None
            finally:
                exc.close()
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            raise TransientModelDownloadError(
                f"Cannot reach the License Server ({exc.__class__.__name__})"
            ) from None
        except json.JSONDecodeError:
            raise ModelDownloadError("License Server returned invalid JSON") from None
        if not isinstance(body, dict):
            raise ModelDownloadError("License Server returned an invalid response")
        return body

    def read_small(self, model_id: str, relative_path: str, *, maximum: int = 1024 * 1024) -> bytes:
        """Read a bounded model metadata object through a short-lived direct URL."""
        access = self.access(model_id, relative_path)
        if access.size > maximum:
            raise ModelDownloadError("model metadata file is too large")
        parsed = urllib.parse.urlparse(access.url)
        loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.username
            or parsed.password
            or (parsed.scheme != "https" and not loopback)
        ):
            raise ModelDownloadError("download URL must use an HTTPS origin")
        request = urllib.request.Request(access.url, headers=access.required_headers)  # noqa: S310
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
                body = response.read(maximum + 1)
        except (OSError, urllib.error.URLError, TimeoutError):
            raise TransientModelDownloadError("cannot read model metadata") from None
        if len(body) != access.size or len(body) > maximum:
            raise ModelDownloadError("invalid model metadata size")
        return body

    def access(self, model_id: str, relative_path: str) -> ModelAccess:
        model = self._model_id(model_id)
        path = self._relative_path(relative_path)
        body = self._json(f"/v1/models/{self._encoded(model)}/access/{self._encoded(path)}")
        try:
            response_path = body["path"]
            url = body["url"]
            size = body["size"]
            source_id = body["source_id"]
            expires_at = body["expires_at"]
            checksums = body.get("checksums", {})
            headers = body.get("required_headers", {})
            if checksums is None:
                checksums = {}
            if headers is None:
                headers = {}
            if (
                response_path != path
                or not isinstance(url, str)
                or not url
                or not isinstance(size, int)
                or isinstance(size, bool)
                or size < 0
                or not isinstance(source_id, str)
                or not source_id
                or not isinstance(expires_at, str)
                or not isinstance(checksums, dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in checksums.items())
                or not isinstance(headers, dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items())
            ):
                raise ValueError
            return ModelAccess(
                path=response_path,
                url=url,
                size=size,
                source_id=source_id,
                expires_at=expires_at,
                checksums=checksums,
                required_headers=headers,
                etag=str(body["etag"]) if body.get("etag") is not None else None,
                version_id=(
                    str(body["version_id"]) if body.get("version_id") is not None else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelDownloadError("License Server returned invalid model metadata") from exc

    def list_models(self, *, limit: int = 1000) -> dict[str, Any]:
        models: set[str] = set()
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            query: dict[str, str | int] = {"limit": limit}
            if cursor:
                query["cursor"] = cursor
            page = self._json(f"/v1/models?{urllib.parse.urlencode(query)}")
            values = page.get("models", [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ModelDownloadError("License Server returned an invalid model listing")
            models.update(self._model_id(value) for value in values)
            cursor_value = page.get("next_cursor")
            cursor = str(cursor_value) if cursor_value else None
            if not cursor:
                break
            if cursor in seen_cursors:
                raise ModelDownloadError("License Server repeated a pagination cursor")
            seen_cursors.add(cursor)
        return {"models": sorted(models), "next_cursor": None}

    def list_page(
        self,
        model_id: str,
        prefix: str = "",
        *,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        model = self._model_id(model_id)
        query: dict[str, str | int] = {"limit": limit}
        if prefix:
            query["prefix"] = self._relative_path(prefix, prefix=True)
        if cursor:
            query["cursor"] = cursor
        return self._json(
            f"/v1/models/{self._encoded(model)}/files?{urllib.parse.urlencode(query)}"
        )

    def list_all(self, model_id: str, prefix: str = "", *, limit: int = 1000) -> dict[str, Any]:
        objects: list[dict[str, Any]] = []
        prefixes: set[str] = set()
        cursor: str | None = None
        model = self._model_id(model_id)
        normalized_prefix = self._relative_path(prefix, prefix=True) if prefix else ""
        seen_cursors: set[str] = set()
        while True:
            page = self.list_page(model, normalized_prefix, limit=limit, cursor=cursor)
            page_objects = page.get("objects", [])
            page_prefixes = page.get("prefixes", [])
            if page_objects is None:
                page_objects = []
            if page_prefixes is None:
                page_prefixes = []
            if not isinstance(page_objects, list) or not isinstance(page_prefixes, list):
                raise ModelDownloadError("License Server returned an invalid listing")
            objects.extend(page_objects)
            prefixes.update(str(item) for item in page_prefixes)
            cursor_value = page.get("next_cursor")
            cursor = str(cursor_value) if cursor_value else None
            if not cursor:
                break
            if cursor in seen_cursors:
                raise ModelDownloadError("License Server repeated a pagination cursor")
            seen_cursors.add(cursor)
        return {
            "model_id": model,
            "prefix": normalized_prefix,
            "objects": objects,
            "prefixes": sorted(prefixes),
            "next_cursor": None,
        }
