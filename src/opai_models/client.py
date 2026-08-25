"""Async License Server client for model metadata and renewable download access."""

from __future__ import annotations

import inspect
import json
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import httpx

from opai_models.errors import extract_http_error_detail, sanitize_error_detail

AsyncLicenseProvider = Callable[[], Awaitable[str]]


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


class _AsyncLicenseTransport:
    """Internal native-async License Server transport."""

    def __init__(
        self,
        api_url: str,
        license_provider: AsyncLicenseProvider,
        timeout: float = 30,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
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
        if not inspect.iscoroutinefunction(license_provider):
            raise TypeError("license_provider must be an async callable")
        self.api_url = api_url.rstrip("/")
        self.license_provider = license_provider
        self.timeout = timeout
        self._owns_http_client = http_client is None
        self.http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,
            headers={"Accept": "application/json", "User-Agent": "opai-models/0.1.0"},
        )

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self.http.aclose()

    async def __aenter__(self) -> _AsyncLicenseTransport:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def _license_key(self) -> str:
        key = await self.license_provider()
        if not isinstance(key, str) or not key:
            raise ModelDownloadError("license key must not be empty")
        return key

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

    async def _json(self, path: str) -> dict[str, Any]:
        try:
            response = await self.http.get(
                self.api_url + path,
                headers={"Authorization": f"Bearer {await self._license_key()}"},
            )
        except httpx.RequestError as exc:
            detail = sanitize_error_detail(exc)
            raise TransientModelDownloadError(
                f"Cannot reach the License Server ({exc.__class__.__name__}): {detail}"
            ) from None
        if response.status_code >= 400:
            error_type = (
                TransientModelDownloadError
                if response.status_code in {408, 429, 500, 502, 503, 504}
                else ModelDownloadError
            )
            message = f"License Server returned HTTP {response.status_code}"
            if detail := extract_http_error_detail(response.content):
                message = f"{message}: {detail}"
            raise error_type(message)
        try:
            body = response.json()
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            raise ModelDownloadError("License Server returned invalid JSON") from None
        if not isinstance(body, dict):
            raise ModelDownloadError("License Server returned an invalid response")
        return body

    async def read_small(
        self, model_id: str, relative_path: str, *, maximum: int = 1024 * 1024
    ) -> bytes:
        """Read a bounded model metadata object through a short-lived direct URL."""
        access = await self.access(model_id, relative_path)
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
        try:
            async with self.http.stream(
                "GET", access.url, headers=access.required_headers, timeout=self.timeout
            ) as response:
                if response.status_code >= 400:
                    message = f"storage returned HTTP {response.status_code} for model metadata"
                    if detail := extract_http_error_detail(await response.aread()):
                        message = f"{message}: {detail}"
                    raise TransientModelDownloadError(message)
                body = bytearray()
                async for block in response.aiter_bytes():
                    body.extend(block)
                    if len(body) > maximum:
                        raise ModelDownloadError("invalid model metadata size")
        except httpx.RequestError as exc:
            detail = sanitize_error_detail(exc)
            raise TransientModelDownloadError(
                f"cannot read model metadata ({exc.__class__.__name__}): {detail}"
            ) from None
        if len(body) != access.size:
            raise ModelDownloadError(
                f"invalid model metadata size: expected {access.size} bytes "
                f"but received {len(body)}"
            )
        return bytes(body)

    async def access(self, model_id: str, relative_path: str) -> ModelAccess:
        model = self._model_id(model_id)
        path = self._relative_path(relative_path)
        body = await self._json(f"/v1/models/{self._encoded(model)}/access/{self._encoded(path)}")
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

    async def list_models(self, *, limit: int = 1000) -> dict[str, Any]:
        models: set[str] = set()
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while True:
            query: dict[str, str | int] = {"limit": limit}
            if cursor:
                query["cursor"] = cursor
            page = await self._json(f"/v1/models?{urllib.parse.urlencode(query)}")
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

    async def list_page(
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
        return await self._json(
            f"/v1/models/{self._encoded(model)}/files?{urllib.parse.urlencode(query)}"
        )

    async def list_all(
        self, model_id: str, prefix: str = "", *, limit: int = 1000
    ) -> dict[str, Any]:
        objects: list[dict[str, Any]] = []
        prefixes: set[str] = set()
        cursor: str | None = None
        model = self._model_id(model_id)
        normalized_prefix = self._relative_path(prefix, prefix=True) if prefix else ""
        seen_cursors: set[str] = set()
        while True:
            page = await self.list_page(model, normalized_prefix, limit=limit, cursor=cursor)
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
