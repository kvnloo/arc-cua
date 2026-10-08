"""Computer use with OpenAI's Decisions API, driven by an agent.

    pip install -e '.[browser]'
    export OPENAI_API_KEY=...
    python examples/decisions_showcase.py --goal "<the job, as the agent was given it>"

Chrome opens on the left and a decision panel on the right, and the script
listens on a local port (printed at start). An agent hands it one subtask at a
time:

    curl -s localhost:PORT/subtask -d '{"title": "...", "url": "https://...",
        "subtask": {"goal": "...", "inputs": {}, "verification": ["..."]}}'

`url` is optional: without it the subtask continues on the current page. Every
action is chosen by gpt-6-luna through POST /v1/decisions; the panel shows each
answer's probabilities, its latency and the running cost as it arrives. The
reply reports how the subtask ended, the page it ended on, and a screenshot path,
so the agent can decide what to hand off next.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import os
import re
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from arc_cua import DesktopExecutor, subtask_from_dict
from arc_cua.backends import ChromeBackend
from arc_cua.backends.cdp import ChromeProcess
from arc_cua.policies import ChoicePolicy, OpenAIDecisionsTransport

ROOT = Path(__file__).resolve().parent.parent
PRICE_PER_INPUT_TOKEN = 0.10 / 1_000_000
REGULAR_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36"
)


# ---- live panel --------------------------------------------------------------------


class Panel:
    """State the panel page polls; updated as subtasks run."""

    def __init__(self, model: str, job: str) -> None:
        self.lock = threading.Lock()
        self.state: dict[str, Any] = {
            "model": model, "job": job, "phase": "ready", "current": None,
            "tasks": [], "latencies": [], "tokens": 0,
        }
        self.update(lambda s: None)

    def update(self, fn: Callable[[dict[str, Any]], None]) -> None:
        with self.lock:
            fn(self.state)
            latencies = self.state["latencies"]
            self.state["summary"] = {
                "decisions": len(latencies),
                "median_ms": round(statistics.median(latencies)) if latencies else None,
                "tokens": self.state["tokens"],
                "cost": self.state["tokens"] * PRICE_PER_INPUT_TOKEN,
            }

    def snapshot(self) -> bytes:
        with self.lock:
            return json.dumps(self.state).encode()


def screen_size() -> tuple[int, int]:
    try:
        out = subprocess.run(
            ["osascript", "-e", 'tell application "Finder" to get bounds of window of desktop'],
            capture_output=True, text=True, timeout=5,
        ).stdout
        _, _, width, height = (int(n) for n in re.findall(r"-?\d+", out))
        return width, height
    except (OSError, ValueError, subprocess.SubprocessError):
        return 1512, 982


# ---- one decision, as the panel shows it ---------------------------------------------


def distribution(answer: dict[str, Any] | None, label: Callable[[str], str], top: int = 4) -> list[list[Any]]:
    probabilities = (answer or {}).get("probabilities") or {}
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)[:top]
    return [[label(key), round(value, 3)] for key, value in ranked]


def describe(decision: Any, snapshot: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    answers = decision.raw.get("answers", {})

    def element(element_id: str) -> str:
        found = snapshot.element(element_id)
        if found is None:
            return element_id
        return found.name or (found.value if isinstance(found.value, str) else "") or found.role or element_id

    action = decision.kind.value if decision.kind else decision.terminal.value
    step: dict[str, Any] = {
        "action": action,
        "latency_ms": decision.latency_ms,
        "confidence": decision.confidence,
        "operation": distribution(answers.get("operation"), lambda key: key),
    }
    if decision.target_id:
        step["target"] = element(decision.target_id)
        step["targets"] = distribution(answers.get(f"{action.lower()}_target"), element, top=3)
    if decision.input_key:
        step["value"] = inputs.get(decision.input_key)
    for field in ("key", "hotkey", "scroll_direction"):
        if getattr(decision, field):
            step["detail"] = getattr(decision, field)
    if decision.click_modifier and decision.click_modifier != "NONE":
        step["detail"] = f"{decision.click_modifier}-click"
    if decision.reason:
        step["reason"] = decision.reason
    return step


def tokens_used(raw: dict[str, Any]) -> int:
    total = 0
    for part in (raw, raw.get("image_verification") or {}):
        usage = part.get("usage") or {}
        total += int(usage.get("input_tokens") or 0)
    return total


# ---- running subtasks --------------------------------------------------------------


class Runner:
    """Runs the agent's subtasks one at a time in one browser tab."""

    def __init__(self, backend: ChromeBackend, transport: OpenAIDecisionsTransport, panel: Panel,
                 screenshots: bool, out_dir: Path) -> None:
        self.backend, self.transport, self.panel = backend, transport, panel
        self.screenshots, self.out_dir = screenshots, out_dir
        self.lock = threading.Lock()

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            return self._run(request)

    def _run(self, request: dict[str, Any]) -> dict[str, Any]:
        subtask = subtask_from_dict(request["subtask"])
        index = len(self.panel.state["tasks"])
        title = request.get("title") or subtask.goal

        def start(s: dict[str, Any]) -> None:
            s["tasks"].append({"title": title, "goal": subtask.goal, "status": "running", "steps": [],
                               "elapsed": None})
            s.update(phase="running", current=index)

        self.panel.update(start)
        if request.get("url"):
            self.backend.navigate(request["url"])
        executor = DesktopExecutor(
            self.backend, ChoicePolicy(self.transport, screenshot_steps=self.screenshots),
        )
        decided_on = self.backend.observe()
        started = time.perf_counter()
        result, error, steps = None, None, []
        try:
            for event in executor.run_iter(subtask):
                step = describe(event.decision, decided_on, dict(subtask.inputs))
                step["at"] = round(time.perf_counter() - started, 2)
                steps.append(step)
                used, latency = tokens_used(event.decision.raw), event.decision.latency_ms

                def record(s: dict[str, Any], step: dict = step, used: int = used, latency: Any = latency) -> None:
                    s["tasks"][index]["steps"].append(step)
                    s["tokens"] += used
                    if latency:
                        s["latencies"].append(latency)

                self.panel.update(record)
                print(f"{step['at']:6.2f}s  {step['action']:<16} {str(step.get('target', ''))[:40]!r}  "
                      f"{step.get('value') or step.get('detail') or ''}  {latency} ms")
                decided_on = event.snapshot
                result = event.result or result
        except Exception as exc:  # A provider error ends this subtask; the agent decides what next.
            error = f"{type(exc).__name__}: {exc}"
        elapsed = round(time.perf_counter() - started, 1)
        status = result.status.value if result else "ERROR"
        self.panel.update(lambda s: s["tasks"][index].update(
            status="verified" if status == "SUBTASK_COMPLETE" else "failed", elapsed=elapsed))

        page = self.backend.observe()
        screenshot = self.out_dir / f"subtask{index + 1}.png"
        png = self.backend._call("Page.captureScreenshot", {"format": "png"})["data"]  # noqa: SLF001
        screenshot.write_bytes(base64.b64decode(png))
        return {
            "status": status,
            "reason": (result.reason if result else None) or error,
            "elapsed_s": elapsed,
            "actions": [f"{s['action']} {s.get('target') or s.get('value') or s.get('detail') or ''}".strip()
                        for s in steps],
            "url": page.context.get("url"),
            "title": page.context.get("title"),
            "screenshot": str(screenshot),
        }


def serve(panel: Panel, runner_ready: threading.Event, runner_box: dict[str, Runner]) -> http.server.HTTPServer:
    page = (Path(__file__).with_name("decisions_showcase.html")).read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body: bytes, kind: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/state"):
                self._send(panel.snapshot(), "application/json")
            else:
                self._send(page, "text/html; charset=utf-8")

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/subtask"):
                self._send(b"{}", "application/json", 404)
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                runner_ready.wait()
                reply = runner_box["runner"].run(request)
                self._send(json.dumps(reply).encode(), "application/json")
            except (ValueError, KeyError, TypeError) as exc:
                self._send(json.dumps({"error": str(exc)}).encode(), "application/json", 400)

        def log_message(self, *_: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---- main ----------------------------------------------------------------------------


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        match = re.match(r"\s*(?:export\s+)?(OPENAI_API_KEY)\s*=\s*['\"]?([^'\"\s]+)", line)
        if match and not os.environ.get(match.group(1)):
            os.environ[match.group(1)] = match.group(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--goal", default="", help="the job, shown at the top of the panel")
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--no-screenshots", action="store_true",
                        help="decide from the page's elements only, without a screenshot per decision")
    parser.add_argument("--headless", action="store_true", help="no windows; the panel is not shown")
    parser.add_argument("--out-dir", type=Path, help="where end-of-subtask screenshots go")
    args = parser.parse_args()
    load_env()

    transport = OpenAIDecisionsTransport()
    panel = Panel(transport.model, args.goal)
    ready, box = threading.Event(), {}
    server = serve(panel, ready, box)
    width, height = screen_size()
    browser_width = width - args.panel_width
    panel_window = None if args.headless else ChromeProcess(
        window_size=(args.panel_width, height),
        extra_args=(f"--app=http://127.0.0.1:{server.server_port}/", f"--window-position={browser_width},0"),
    )
    browser_args: tuple[str, ...] = ("--window-position=0,0",)
    if args.headless:
        # Headless Chrome names itself in its user agent; present as the regular browser.
        browser_args += (f"--user-agent={REGULAR_USER_AGENT}",)
    out_dir = args.out_dir or Path(tempfile.mkdtemp(prefix="arc-showcase-"))
    out_dir.mkdir(parents=True, exist_ok=True)
    transport.warm()
    try:
        with ChromeBackend.launch(
            headless=args.headless, window_size=(browser_width, height), extra_args=browser_args,
            capture_screenshots=not args.no_screenshots,
        ) as backend:
            box["runner"] = Runner(backend, transport, panel, not args.no_screenshots, out_dir)
            ready.set()
            print(f"Ready: POST subtasks to http://127.0.0.1:{server.server_port}/subtask", flush=True)
            threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if panel_window:
            panel_window.close()
        server.shutdown()


if __name__ == "__main__":
    main()
