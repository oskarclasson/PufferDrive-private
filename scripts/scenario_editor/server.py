"""Browser driving workbench: draw road maps, place actors, goals, obstacles and light timings, then evaluate.

Usage (from the repo root, venv active):
    python scripts/scenario_editor/server.py --map-dir data/nuplan_mini_val --drawn-map-dir data/drawn_maps \\
        --model weights/k_scaled_0036_1000/final_model.pt --port 8090
Then open http://localhost:8090 (forward the port over ssh when running remotely). The Map tab draws road
layouts into --drawn-map-dir; the Scenario tab places traffic on any map from either directory.
"""

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import importlib.util

import yaml

import scenario_builder

from pufferlib.ocean.drive import route_graph

EDITOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = EDITOR_DIR.parents[1]
MAP_EDITOR_DIR = REPO_ROOT / "scripts" / "map_editor"
MAP_EDITOR_PREFIX = "/map-editor/"
SCENARIO_EDITOR_PREFIX = "/scenario/"
DRAWN_BIN_PREFIX = "drawn__"  # map editor BIN_PREFIX
BENCHMARK_TEMPLATE_PATH = EDITOR_DIR / "benchmark_template.yaml"
BENCHMARK_NAME = "scenario_editor"
SCENARIO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
RUN_LABEL_PATTERN = re.compile(r"^(?P<scenario>[A-Za-z0-9_-]{1,64})_(?P<stamp>\d{8}-\d{6})$")
RUN_STAMP_FORMAT = "%Y%m%d-%H%M%S"
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAP_CACHE_SIZE = 8
LOG_TAIL_LINES = 60
MPPI_SAMPLE_COUNT = 256
PREDICT_HORIZON_SECONDS = 10.0
EVAL_AGENT_SLOTS = 300
MIN_GOAL_RADIUS_M = 0.5
MAX_GOAL_RADIUS_M = 20.0
MIN_INFERRED_GOAL_SPACING_M = 2.0
MAX_INFERRED_GOAL_SPACING_M = 50.0
SUMMARY_METRIC_KEYS = (
    "active_agent_count",
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
    "stop_sign_violation_rate",
    "num_goals_reached",
    "dnf_rate",
)


def load_map_editor_module():
    """The map editor's own server module, loaded under a distinct name (both files are called server.py)."""
    sys.path.insert(0, str(MAP_EDITOR_DIR))
    module_spec = importlib.util.spec_from_file_location("map_editor_server", MAP_EDITOR_DIR / "server.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def find_eval_pid(run_label):
    """Pid of the live `puffer eval` writing this run, found via /proc so runs survive server restarts."""
    marker = f"eval.output_name={run_label}".encode()
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            arguments = (proc_dir / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if marker in arguments:
            return int(proc_dir.name)
    return None


def read_log_tail(log_path):
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="replace").replace("\r", "\n")
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-LOG_TAIL_LINES:])


class EditorState:
    def __init__(self, args):
        self.map_dirs = [Path(args.drawn_map_dir).resolve(), Path(args.map_dir).resolve()]  # drawn maps list first
        self.scenario_dir = Path(args.scenario_dir).resolve()
        self.model_path = Path(args.model).resolve()
        self.eval_root = self.model_path.parent / "eval"
        self.puffer_executable = shutil.which("puffer") or str(Path(sys.executable).parent / "puffer")
        self.map_cache = {}
        self.lock = threading.Lock()

    def map_paths(self):
        """Every base map by file name; re-scanned per request so maps saved from the Map tab show up at once."""
        paths = {}
        for map_dir in self.map_dirs:
            for path in sorted(map_dir.glob("*.bin")):
                paths.setdefault(path.name, path)
        return paths

    def map_path(self, map_name):
        path = self.map_paths().get(map_name)
        if path is None:
            raise KeyError(map_name)
        return path

    def map_geometry(self, map_name):
        path = self.map_path(map_name)
        cache_key = (map_name, path.stat().st_mtime_ns)
        with self.lock:
            if cache_key in self.map_cache:
                return self.map_cache[cache_key]
        geometry = scenario_builder.export_map_geometry(path)
        with self.lock:
            if len(self.map_cache) >= MAP_CACHE_SIZE:
                self.map_cache.pop(next(iter(self.map_cache)))
            self.map_cache[cache_key] = geometry
        return geometry

    def route_goals(self, request):
        """Goals spaced along the shortest lane route from an actor's pose to a destination point."""
        map_name = str(request.get("map", ""))
        path = self.map_path(map_name)
        number = scenario_builder._finite
        start = [number(request.get(key), key) for key in ("x", "y", "heading")]
        destination = request.get("destination")
        if not isinstance(destination, list) or len(destination) != 2:
            raise scenario_builder.ScenarioSpecError("destination must be [x, y]")
        destination = [number(value, "destination") for value in destination]
        spacing_m = number(
            request.get("spacing_m"),
            "spacing_m",
            scenario_builder.MIN_GOAL_SPACING_M,
            scenario_builder.MAX_GOAL_SPACING_M,
        )
        cache_key = ("route", map_name, path.stat().st_mtime_ns)
        with self.lock:
            graph = self.map_cache.get(cache_key)
        try:
            if graph is None:
                graph = route_graph.RouteGraph(scenario_builder.read_bin(path)["roads"])
                with self.lock:
                    if len(self.map_cache) >= MAP_CACHE_SIZE:
                        self.map_cache.pop(next(iter(self.map_cache)))
                    self.map_cache[cache_key] = graph
            return graph.route_goals(*start, *destination, spacing_m, scenario_builder.MAX_GOALS_PER_ACTOR)
        except route_graph.RouteError as error:
            raise scenario_builder.ScenarioSpecError(str(error)) from error

    def map_origin(self, map_name):
        """Lat/lon of (0, 0) for maps drawn over an .osm backdrop; None for maps without a geographic origin."""
        path = self.map_path(map_name)
        if path.parent != self.map_dirs[0] or not path.name.startswith(DRAWN_BIN_PREFIX):
            return None
        spec_path = path.parent / "specs" / f"{path.stem[len(DRAWN_BIN_PREFIX):]}.json"
        if not spec_path.is_file():
            return None
        origin = json.loads(spec_path.read_text()).get("origin")
        if not isinstance(origin, dict) or not all(isinstance(origin.get(k), (int, float)) for k in ("lat", "lon")):
            return None
        return {"lat": float(origin["lat"]), "lon": float(origin["lon"])}

    def scenario_paths(self, scenario_name):
        if not SCENARIO_NAME_PATTERN.match(scenario_name):
            raise scenario_builder.ScenarioSpecError("name must be 1-64 characters of letters, digits, _ or -")
        root = self.scenario_dir / scenario_name
        return root, root / "spec.json", root / "map" / f"custom__{scenario_name}.bin"

    def save_scenario(self, spec):
        scenario_name = str(spec.get("name", ""))
        root, spec_path, bin_path = self.scenario_paths(scenario_name)
        map_name = spec.get("map")
        map_path = self.map_paths().get(map_name)
        if map_path is None:
            raise scenario_builder.ScenarioSpecError(f"unknown map {map_name!r}")
        scenario_steps = scenario_builder.build_scenario_bin(map_path, spec, bin_path, scenario_name)
        spec_path.write_text(json.dumps(spec, indent=1))
        return scenario_steps

    def list_scenarios(self):
        if not self.scenario_dir.exists():
            return []
        return sorted(p.parent.name for p in self.scenario_dir.glob("*/spec.json"))

    def start_eval(self, request):
        scenario_name = str(request.get("name", ""))
        root, spec_path, bin_path = self.scenario_paths(scenario_name)
        if not spec_path.exists() or not bin_path.exists():
            raise scenario_builder.ScenarioSpecError("save the scenario before evaluating it")
        goal_radius = request.get("goal_radius", 2.0)
        if not isinstance(goal_radius, (int, float)) or not MIN_GOAL_RADIUS_M <= goal_radius <= MAX_GOAL_RADIUS_M:
            raise scenario_builder.ScenarioSpecError("goal_radius out of range")
        inferred_goal_spacing_m = request.get("inferred_goal_spacing_m", 10.0)
        if (
            not isinstance(inferred_goal_spacing_m, (int, float))
            or not MIN_INFERRED_GOAL_SPACING_M <= inferred_goal_spacing_m <= MAX_INFERRED_GOAL_SPACING_M
        ):
            raise scenario_builder.ScenarioSpecError(
                f"inferred goal spacing must be {MIN_INFERRED_GOAL_SPACING_M:g}-{MAX_INFERRED_GOAL_SPACING_M:g} m"
            )
        if request.get("tts") and request.get("mppi", True):
            raise scenario_builder.ScenarioSpecError("choose MPPI or test-time search, not both")
        with self.lock:
            if any(run["status"] == "running" for run in self.list_runs()):
                raise scenario_builder.ScenarioSpecError("an evaluation is already running (see Results)")

            spec = json.loads(spec_path.read_text())
            # Actors are written first and in order, so an actor's index is its scenario agent id.
            ego_actor_idx = next((idx for idx, actor in enumerate(spec["actors"]) if actor.get("ego")), None)
            scenario_steps = int(round(float(spec["duration_s"]) / scenario_builder.SIM_DT_SECONDS))
            benchmark = yaml.safe_load(BENCHMARK_TEMPLATE_PATH.read_text())
            (entry,) = [b for b in benchmark["benchmarks"] if b["name"] == BENCHMARK_NAME]
            entry["env"]["scenario_length"] = scenario_steps
            benchmark_path = root / "benchmark.yaml"
            benchmark_path.write_text(yaml.safe_dump(benchmark, sort_keys=False))

            run_label = f"{scenario_name}_{datetime.datetime.now():{RUN_STAMP_FORMAT}}"
            settings = {
                "mppi": bool(request.get("mppi", True)),
                "tts": bool(request.get("tts", False)),
                "ego_actor": "none" if ego_actor_idx is None else f"A{ego_actor_idx + 1}",
                "goal_inference": bool(request.get("goal_inference", False) or request.get("joint_prediction", False)),
                "joint_prediction": bool(request.get("joint_prediction", False)),
                "prediction": bool(request.get("prediction", True)),
                "capture_observations": bool(request.get("capture_observations", True)),
                "goal_radius": float(goal_radius),
                "inferred_goal_spacing_m": float(inferred_goal_spacing_m),
                "duration_s": float(spec["duration_s"]),
                "map": spec["map"],
                "actor_count": len(spec["actors"]),
                "obstacle_count": len(spec.get("obstacles", [])),
            }
            command = [
                self.puffer_executable,
                "eval",
                "puffer_drive",
                BENCHMARK_NAME,
                f"load_model_path={self.model_path}",
                f"eval.benchmark_config={benchmark_path}",
                f"eval.map_dir={bin_path.parent}",
                "eval.num_scenarios=1",
                f"eval.num_agents={EVAL_AGENT_SLOTS}",
                "eval.goal_regen_mode=rolling",
                f"eval.goal_radius={settings['goal_radius']}",
                f"eval.goal_inference_goal_spacing_m={settings['inferred_goal_spacing_m']}",
                "eval.disable_red_light_infractions=false",
                "eval.disable_stop_sign_infractions=false",
                "eval.render_scenarios=true",
                f"eval.capture_observations={str(settings['capture_observations']).lower()}",
                f"eval.predict_horizon_s={PREDICT_HORIZON_SECONDS if settings['prediction'] else 0.0}",
                f"eval.mppi_samples={MPPI_SAMPLE_COUNT if settings['mppi'] else 0}",
                f"eval.tts={str(settings['tts']).lower()}",
                f"eval.ego_agent_id={'null' if ego_actor_idx is None else ego_actor_idx}",
                f"eval.goal_inference={str(settings['goal_inference']).lower()}",
                f"eval.joint_prediction={str(settings['joint_prediction']).lower()}",
                f"eval.output_name={run_label}",
                "wandb=False",
            ]
            # Own session so a Ctrl-C on the editor server doesn't kill evaluations in flight.
            (root / f"eval_{run_label}.json").write_text(json.dumps(settings, indent=1))
            with open(root / f"eval_{run_label}.log", "w") as log_file:
                subprocess.Popen(
                    command, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True
                )
        return run_label

    def run_paths(self, run_label):
        match = RUN_LABEL_PATTERN.match(run_label)
        if not match:
            raise KeyError(run_label)
        root = self.scenario_dir / match["scenario"]
        return root / f"eval_{run_label}.log", root / f"eval_{run_label}.json"

    def run_result_dir(self, run_label):
        output_root = self.eval_root / f"{BENCHMARK_NAME}_{run_label}"
        if not output_root.exists():
            return None
        finished = sorted(p.parent for p in output_root.glob("*/evaluation_summary.json"))
        return finished[-1] if finished else None

    def describe_run(self, log_path):
        run_label = log_path.stem[len("eval_") :]
        match = RUN_LABEL_PATTERN.match(run_label)
        if not match:
            return None
        settings_path = log_path.with_suffix(".json")
        settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
        started = datetime.datetime.strptime(match["stamp"], RUN_STAMP_FORMAT)
        result_dir = self.run_result_dir(run_label)
        pid = None if result_dir else find_eval_pid(run_label)
        status = "done" if result_dir else ("running" if pid else "failed")
        if status == "failed" and log_path.with_suffix(".stopped").exists():
            status = "stopped"
        run = {
            "label": run_label,
            "scenario": match["scenario"],
            "started": started.isoformat(sep=" "),
            "status": status,
            "settings": settings,
            "metrics": None,
            "replay_url": None,
        }
        if status == "running":
            run["elapsed_s"] = (datetime.datetime.now() - started).total_seconds()
        if result_dir is None:
            return run
        summary = json.loads((result_dir / "evaluation_summary.json").read_text())["metrics_mean"]
        run["metrics"] = summary
        replays = sorted((result_dir / "rendered_replays").glob("custom__*.html"))
        if replays:
            run["replay_url"] = "/results/" + replays[0].relative_to(self.eval_root).as_posix()
        return run

    def list_runs(self):
        if not self.scenario_dir.exists():
            return []
        runs = [self.describe_run(p) for p in self.scenario_dir.glob("*/eval_*.log")]
        runs = [r for r in runs if r is not None]
        for run in runs:
            if run["metrics"] is not None:
                run["metrics"] = {k: run["metrics"].get(k) for k in SUMMARY_METRIC_KEYS}
        return sorted(runs, key=lambda r: r["started"], reverse=True)

    def run_detail(self, run_label):
        log_path, _ = self.run_paths(run_label)
        if not log_path.exists():
            raise KeyError(run_label)
        run = self.describe_run(log_path)
        run["log"] = read_log_tail(log_path)
        return run

    def stop_run(self, run_label):
        log_path, _ = self.run_paths(run_label)
        pid = find_eval_pid(run_label)
        if pid is None:
            raise scenario_builder.ScenarioSpecError("run is not running")
        log_path.with_suffix(".stopped").touch()
        os.killpg(os.getpgid(pid), signal.SIGTERM)


def make_handler(state, map_editor_state, map_spec_error):
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

        def _send_file(self, path):
            content_type = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}.get(
                path.suffix, "application/octet-stream"
            )
            self._send(200, path.read_bytes(), content_type)

        def do_GET(self):
            url = urlparse(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/index.html"):
                    return self._send_file(EDITOR_DIR / "app.html")
                if url.path == SCENARIO_EDITOR_PREFIX:
                    return self._send_file(EDITOR_DIR / "editor.html")
                if url.path == MAP_EDITOR_PREFIX:
                    return self._send_file(MAP_EDITOR_DIR / "editor.html")
                if url.path == MAP_EDITOR_PREFIX + "basemap.js":
                    return self._send_file(MAP_EDITOR_DIR / "basemap.js")
                if url.path == MAP_EDITOR_PREFIX + "api/maps":
                    return self._send(
                        200, {"maps": map_editor_state.list_maps(), "out_dir": str(map_editor_state.out_dir)}
                    )
                if url.path == MAP_EDITOR_PREFIX + "api/map":
                    spec_path, _ = map_editor_state.paths(query.get("name", ""))
                    return self._send(200, json.loads(spec_path.read_text()))
                if url.path == "/api/maps":
                    return self._send(200, {"maps": list(state.map_paths()), "map_dirs": [str(d) for d in state.map_dirs]})
                if url.path == "/api/map":
                    map_name = query.get("name", "")
                    return self._send(200, {**state.map_geometry(map_name), "origin": state.map_origin(map_name)})
                if url.path == "/api/scenarios":
                    return self._send(200, {"scenarios": state.list_scenarios()})
                if url.path == "/api/scenario":
                    _, spec_path, _ = state.scenario_paths(query.get("name", ""))
                    return self._send(200, json.loads(spec_path.read_text()))
                if url.path == "/api/runs":
                    return self._send(200, {"runs": state.list_runs()})
                if url.path == "/api/run":
                    return self._send(200, state.run_detail(query.get("label", "")))
                if url.path.startswith("/results/"):
                    target = (state.eval_root / unquote(url.path[len("/results/") :])).resolve()
                    if state.eval_root not in target.parents or not target.is_file():
                        return self._send(404, {"error": "not found"})
                    return self._send_file(target)
            except (KeyError, FileNotFoundError):
                return self._send(404, {"error": "not found"})
            except (scenario_builder.ScenarioSpecError, map_spec_error) as error:
                return self._send(400, {"error": str(error)})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_REQUEST_BYTES:
                return self._send(413, {"error": "request too large or empty"})
            try:
                body = json.loads(self.rfile.read(length))
                if url.path == MAP_EDITOR_PREFIX + "api/save" and isinstance(body, dict):
                    return self._send(200, {"ok": True, **map_editor_state.save(body)})
                if url.path == "/api/scenario":
                    return self._send(200, {"ok": True, "steps": state.save_scenario(body)})
                if url.path == "/api/route" and isinstance(body, dict):
                    return self._send(200, state.route_goals(body))
                if url.path == "/api/evaluate":
                    return self._send(200, {"label": state.start_eval(body)})
                if url.path == "/api/run/stop":
                    state.stop_run(str(body.get("label", "")))
                    return self._send(200, {"ok": True})
            except KeyError:
                return self._send(404, {"error": "not found"})
            except (json.JSONDecodeError, scenario_builder.ScenarioSpecError, map_spec_error) as error:
                return self._send(400, {"error": str(error)})
            return self._send(404, {"error": "not found"})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", default="data/nuplan_mini_val", help="Directory of base map .bin files")
    parser.add_argument("--drawn-map-dir", default="data/drawn_maps", help="Where the Map tab saves drawn maps")
    parser.add_argument("--scenario-dir", default="data/editor_scenarios", help="Where authored scenarios are saved")
    parser.add_argument("--model", default="weights/k_scaled_0036_1000/final_model.pt", help="Policy checkpoint")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (0.0.0.0 exposes it to the network)")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    state = EditorState(args)
    map_editor = load_map_editor_module()
    map_editor_state = map_editor.EditorState(args.drawn_map_dir)
    if not state.model_path.is_file():
        sys.exit(f"model checkpoint not found: {state.model_path}")
    handler = make_handler(state, map_editor_state, map_editor.map_builder.MapSpecError)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Workbench on http://{args.host}:{args.port}  ({len(state.map_paths())} maps, model {state.model_path})")
    server.serve_forever()


if __name__ == "__main__":
    main()
