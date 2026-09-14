"""Concurrent HTTP with a single main-thread physics/render owner."""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from tabletop_vla.planning.task_planner import executable_task, plan
from tabletop_vla.sim.runtime import Simulation


class Engine:
    """Commands, stepping, and OpenGL all execute on the process main thread."""

    def __init__(self):
        self.sim = Simulation()
        self.commands = queue.Queue(maxsize=64)
        self.last_tick = time.perf_counter()
        self.remainder = 0.0

    def execute(self, path, body):
        if path == "state":
            return self.sim.state()
        if path == "frame":
            return self.sim.frame(body.get("camera", "overview"))
        if path == "reset":
            return self.sim.reset(body.get("seed", 0))
        if path == "playback":
            return self.sim.playback(body["running"])
        if path == "demo":
            return self.sim.start_demo(body.get("seed", 0))
        if path == "task":
            return self.sim.start_task(body.get("seed", 0), body.get("name", "cup-place"))
        if path == "step":
            if self.sim.running:
                raise ValueError("Pause physics before using single step")
            return self.sim.step(body.get("steps", 50))
        if path == "control":
            self.sim.control(body["name"], float(body["value"]))
            self.sim.running = True
            return self.sim.state()
        if path == "reach":
            result = self.sim.reach(body["arm"], body["target"])
            if result["last_motion"]["accepted"]:
                self.sim.running = True
            return self.sim.state()
        raise ValueError("Unknown simulation command")

    def tick(self):
        now = time.perf_counter()
        elapsed = min(now - self.last_tick, 0.1)
        self.last_tick = now
        if self.sim.running:
            self.remainder += elapsed
            count = math.floor(self.remainder / self.sim.model.opt.timestep)
            if count:
                self.sim.step(min(count, 500))
                self.remainder -= count * self.sim.model.opt.timestep
        else:
            self.remainder = 0

    def poll(self):
        self.tick()
        try:
            command, body, future = self.commands.get(timeout=0.005)
        except queue.Empty:
            return
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(self.execute(command, body))
        except Exception as exc:  # noqa: BLE001 -- report worker errors to their waiting HTTP request
            future.set_exception(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8771, type=int)
    args = parser.parse_args()
    engine = Engine()

    class Handler(BaseHTTPRequestHandler):
        timeout = 5

        def reply(self, data, mime="application/json", status=200):
            if not isinstance(data, bytes):
                data = json.dumps(data, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def submit(self, name, body):
            future = Future()
            engine.commands.put_nowait((name, body, future))
            try:
                return future.result(timeout=6)
            except TimeoutError:
                future.cancel()
                raise

        def do_GET(self):
            request = urlparse(self.path)
            try:
                if request.path in {"/", "/console.js"}:
                    file = "console.html" if request.path == "/" else "console.js"
                    mime = (
                        "text/html; charset=utf-8" if file.endswith("html") else "text/javascript"
                    )
                    self.reply(Path(__file__).with_name(file).read_bytes(), mime)
                elif request.path == "/api/state":
                    self.reply(self.submit("state", {}))
                elif request.path == "/api/frame":
                    camera = parse_qs(request.query).get("camera", ["overview"])[0]
                    self.reply(self.submit("frame", {"camera": camera}), "image/jpeg")
                else:
                    self.reply({"error": "Not found"}, status=404)
            except (ValueError, KeyError, TypeError) as exc:
                self.reply({"error": str(exc)}, status=400)
            except (RuntimeError, OSError, TimeoutError, queue.Full) as exc:
                self.reply({"error": f"Simulation unavailable: {exc}"}, status=503)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if origin and origin not in {
                f"http://127.0.0.1:{args.port}",
                f"http://localhost:{args.port}",
            }:
                self.reply({"error": "Local console requests only"}, status=403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384 or self.headers.get_content_type() != "application/json":
                    raise ValueError("A small JSON request is required")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise TypeError("Expected a JSON object")
                if self.path == "/api/plan":
                    result = {
                        "backend": "rule_based_preview",
                        "executable": False,
                        "skills": [asdict(skill) for skill in plan(body["instruction"])],
                    }
                elif self.path == "/api/execute":
                    task = executable_task(body["instruction"])
                    result = self.submit("task", {"name": task, "seed": body.get("seed", 0)})
                elif self.path in {
                    "/api/reset",
                    "/api/playback",
                    "/api/demo",
                    "/api/task",
                    "/api/step",
                    "/api/control",
                    "/api/reach",
                }:
                    result = self.submit(self.path.rsplit("/", 1)[1], body)
                else:
                    self.reply({"error": "Not found"}, status=404)
                    return
                self.reply(result)
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                self.reply({"error": str(exc)}, status=400)
            except (RuntimeError, OSError, TimeoutError, queue.Full) as exc:
                self.reply({"error": f"Simulation unavailable: {exc}"}, status=503)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"MuJoCo console: http://127.0.0.1:{args.port}", flush=True)
    try:
        while True:
            engine.poll()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        engine.sim.close()


if __name__ == "__main__":
    main()
