"""Browser map editor: load an OSM / lanelet2 map as a backdrop, draw lanes, road edges and stop lines, save as .bin.

Usage (from the repo root, venv active):
    python scripts/map_editor/server.py --out-dir data/drawn_maps --port 8095
Then open http://localhost:8095. Saved bins go to --out-dir, which works as --map-dir for the scenario editor.
"""

import argparse
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import map_builder

EDITOR_DIR = Path(__file__).resolve().parent
MAP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_REQUEST_BYTES = 32 * 1024 * 1024
BIN_PREFIX = "drawn__"


class EditorState:
    def __init__(self, out_dir):
        self.out_dir = Path(out_dir).resolve()
        self.spec_dir = self.out_dir / "specs"
        self.lock = threading.Lock()

    def paths(self, map_name):
        if not MAP_NAME_PATTERN.match(map_name):
            raise map_builder.MapSpecError("name must be 1-64 characters of letters, digits, _ or -")
        return self.spec_dir / f"{map_name}.json", self.out_dir / f"{BIN_PREFIX}{map_name}.bin"

    def save(self, request):
        map_name = str(request.get("name", ""))
        spec_path, bin_path = self.paths(map_name)
        with self.lock:
            summary = map_builder.build_map_bin(request.get("spec"), bin_path, map_name)
            self.spec_dir.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(request["spec"]))
        return {**summary, "bin_path": str(bin_path)}

    def list_maps(self):
        return sorted(p.stem for p in self.spec_dir.glob("*.json")) if self.spec_dir.exists() else []


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send(self, status, body, content_type="application/json"):
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._send(200, (EDITOR_DIR / "editor.html").read_bytes(), "text/html")
                if url.path == "/basemap.js":
                    return self._send(200, (EDITOR_DIR / "basemap.js").read_bytes(), "text/javascript")
                if url.path == "/api/maps":
                    return self._send(200, {"maps": state.list_maps(), "out_dir": str(state.out_dir)})
                if url.path == "/api/map":
                    spec_path, _ = state.paths(query.get("name", ""))
                    return self._send(200, json.loads(spec_path.read_text()))
            except FileNotFoundError:
                return self._send(404, {"error": "not found"})
            except map_builder.MapSpecError as error:
                return self._send(400, {"error": str(error)})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_REQUEST_BYTES:
                return self._send(413, {"error": "request too large or empty"})
            try:
                body = json.loads(self.rfile.read(length))
                if url.path == "/api/save" and isinstance(body, dict):
                    return self._send(200, {"ok": True, **state.save(body)})
            except (json.JSONDecodeError, map_builder.MapSpecError) as error:
                return self._send(400, {"error": str(error)})
            return self._send(404, {"error": "not found"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/drawn_maps", help="Where saved .bin maps (and their specs) go")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 exposes it to the network)")
    parser.add_argument("--port", type=int, default=8095)
    args = parser.parse_args()
    state = EditorState(args.out_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"Map editor on http://{args.host}:{args.port}  (saving to {state.out_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
