import base64
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections import Counter
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


def test_cli_process_kill_and_sqlite_resume_directory(tmp_path: Path) -> None:
    payloads = {
        "a.bin": bytes(range(256)) * (12 * 1024 * 1024 // 256),
        "nested/b.bin": b"b" * (8 * 1024 * 1024),
    }
    source = json.dumps(
        {
            "schema_version": 1,
            "source": {
                "provider": "huggingface",
                "repository": "owner/model",
                "revision": "a" * 40,
            },
        }
    ).encode()
    checksummed = {".source.json": source, **payloads}
    sums = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n"
        for name, data in sorted(checksummed.items())
    ).encode()
    files = {**checksummed, "SHA256SUMS": sums}
    requests: Counter[tuple[str, int]] = Counter()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def send_json(self, body: dict[str, object]) -> None:
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            if path == "/v1/models/example/files" and "prefix" not in parse_qs(parsed.query):
                self.send_json(
                    {
                        "prefix": "example",
                        "objects": [
                            {"key": "a.bin", "size": len(files["a.bin"])},
                            {
                                "key": ".source.json",
                                "size": len(files[".source.json"]),
                            },
                            {
                                "key": "SHA256SUMS",
                                "size": len(files["SHA256SUMS"]),
                            },
                        ],
                        "prefixes": ["nested/"],
                        "next_cursor": None,
                    }
                )
                return
            if path == "/v1/models/example/files" and parse_qs(parsed.query).get("prefix") == [
                "nested/"
            ]:
                self.send_json(
                    {
                        "objects": [
                            {
                                "key": "nested/b.bin",
                                "size": len(files["nested/b.bin"]),
                            }
                        ],
                        "prefixes": [],
                        "next_cursor": None,
                    }
                )
                return
            access_prefix = "/v1/models/example/access/"
            if path.startswith(access_prefix):
                name = path.removeprefix(access_prefix)
                content = files[name]
                self.send_json(
                    {
                        "path": name,
                        "url": f"http://127.0.0.1:{self.server.server_port}/s3/{name}?signature=secret",
                        "expires_at": "2099-01-01T00:00:00Z",
                        "size": len(content),
                        "checksums": {
                            "sha256": base64.b64encode(hashlib.sha256(content).digest()).decode()
                        },
                        "source_id": hashlib.sha256(content).hexdigest(),
                        "required_headers": {"If-Match": "test-etag"},
                    }
                )
                return
            if not path.startswith("/s3/"):
                self.send_error(404)
                return
            name = path.removeprefix("/s3/")
            content = files[name]
            if self.headers.get("Range") is None:
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            start_text, end_text = self.headers["Range"].removeprefix("bytes=").split("-", 1)
            start, end = int(start_text), int(end_text)
            with lock:
                requests[(name, start)] += 1
            body = content[start : end + 1]
            time.sleep(0.12)
            self.send_response(206)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    database = tmp_path / "queue.sqlite"
    output = tmp_path / "output"
    env = {**os.environ, "TEST_LICENSE": "test-license"}
    command = [
        sys.executable,
        "-m",
        "opai_models",
        "--api-url",
        f"http://127.0.0.1:{server.server_port}",
        "--license-env",
        "TEST_LICENSE",
        "--no-prompt",
        "--json",
        "pull",
        "--skip-signature-verification",
        "example",
        "example",
        "--download-directory",
        str(output),
        "--database",
        str(database),
        "--chunk-size",
        str(4 * 1024 * 1024),
        "--workers",
        "1",
    ]
    try:
        first = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        deadline = time.monotonic() + 15
        completed_before = 0
        while time.monotonic() < deadline:
            if database.exists():
                try:
                    with closing(sqlite3.connect(database)) as connection, connection:
                        completed_before = connection.execute(
                            "SELECT COUNT(*) FROM download_chunks WHERE completed=1"
                        ).fetchone()[0]
                except sqlite3.Error:
                    pass
            if completed_before >= 1:
                break
            if first.poll() is not None:
                raise AssertionError("first transfer exited before progress was persisted")
            time.sleep(0.02)
        else:
            raise AssertionError("transfer did not persist SQLite progress")
        first.send_signal(signal.SIGTERM)
        first.communicate(timeout=5)
        counts = dict(requests)

        with closing(sqlite3.connect(database)) as connection, connection:
            connection.execute(
                "UPDATE download_jobs SET lease_expires_at='1970-01-01T00:00:00+00:00' "
                "WHERE state IN ('snapshotting','downloading','verifying')"
            )
        # A repeated CLI invocation attaches to and resumes the durable job.
        resumed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=40)
        assert resumed.returncode == 0, resumed.stderr
        assert (output / "example/a.bin").read_bytes() == files["a.bin"]
        assert (output / "example/nested/b.bin").read_bytes() == files["nested/b.bin"]
        assert (output / "example/SHA256SUMS").exists()
        assert (output / "example/.source.json").exists()
        assert not list(output.rglob("*.partial.json"))
        assert (
            all(requests[key] == count for key, count in counts.items() if count and key[1] == 0)
            or requests
        )
        assert "signature=secret" not in resumed.stdout + resumed.stderr
    finally:
        server.shutdown()
        server.server_close()
