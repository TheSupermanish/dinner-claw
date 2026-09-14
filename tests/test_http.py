"""Exercise real HTTP handlers; engine-only tests miss request routing failures."""

import json
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest


def test_http_playback_and_task_route():
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "tabletop_vla.server", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )

    def request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(f"http://127.0.0.1:{port}{path}", data=data,
                      headers={"Content-Type": "application/json"})
        return urlopen(req, timeout=8)

    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                with request("/api/state") as response:
                    assert not json.load(response)["running"]
                break
            except URLError:
                if time.monotonic() >= deadline or process.poll() is not None:
                    raise
                time.sleep(0.05)
        # An idle connection must not monopolize the HTTP server.
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            with request("/api/task", {"name": "cup-place", "seed": 3}) as response:
                result = json.load(response)
                assert result["task"]["status"] == "running"
            time.sleep(0.15)
            with request("/api/playback", {"running": False}) as response:
                paused = json.load(response)["time"]
                assert paused > 0.05
            time.sleep(0.05)
            with request("/api/state") as response:
                assert json.load(response)["time"] == paused
            with request("/api/frame") as response:
                assert response.headers["Content-Type"] == "image/jpeg"
                assert response.read().startswith(b"\xff\xd8")
            with request("/") as response:
                assert b"Pick &amp; place cup" in response.read()
            with pytest.raises(HTTPError) as error:
                request("/api/task", {"name": "unimplemented-drawer"})
            assert error.value.code == 400
            with pytest.raises(HTTPError) as error:
                request("/api/execute", {"instruction": "Pick up the cup, then open the drawer"})
            assert error.value.code == 400
            with request("/api/state") as response:
                assert json.load(response)["seed"] == 3
            with request("/api/execute", {
                "instruction": "Pick up the blue cup and place it on its marker.", "seed": 4,
            }) as response:
                assert json.load(response)["task"]["name"] == "camera-cup-place"
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        process.stderr.close()
