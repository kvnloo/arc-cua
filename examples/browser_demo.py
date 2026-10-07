"""Run one arc-cua subtask in Chrome with JEV or OpenAI Decisions.

    pip install -e '.[browser]'
    export TYPESAFE_API_KEY=...
    python examples/browser_demo.py https://en.wikipedia.org \\
        "Open the Wikipedia article about Gödel's incompleteness theorems" \\
        --input "query=Gödel's incompleteness theorems" \\
        --verify "The article titled Gödel's incompleteness theorems is open"

With --provider openai (OPENAI_API_KEY), --screenshots also sends a screenshot
with every decision.

Chrome runs with a temporary profile. Input goes through DevTools, so the
window can stay in the background while the agent works.
"""

from __future__ import annotations

import argparse
import time

from arc_cua import DesktopExecutor, Subtask
from arc_cua.backends import ChromeBackend
from arc_cua.policies import ChoicePolicy, OpenAIDecisionsTransport, TypeSafeJevPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url")
    parser.add_argument("goal")
    parser.add_argument("--input", action="append", default=[], metavar="NAME=VALUE",
                        help="literal value the agent may type or select (repeatable)")
    parser.add_argument("--verify", action="append", required=True, help="completion criterion (repeatable)")
    parser.add_argument("--max-actions", type=int, default=15)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--provider", choices=("jev", "openai"), default="jev")
    parser.add_argument("--screenshots", action="store_true",
                        help="send a screenshot with every decision (openai only)")
    args = parser.parse_args()
    if args.screenshots and args.provider != "openai":
        parser.error("--screenshots needs --provider openai")
    if args.provider == "openai":
        policy = ChoicePolicy(OpenAIDecisionsTransport(), screenshot_steps=args.screenshots)
    else:
        policy = TypeSafeJevPolicy()

    task = Subtask(
        goal=args.goal,
        inputs=dict(item.split("=", 1) for item in args.input),
        verification=tuple(args.verify),
        max_actions=args.max_actions,
    )
    with ChromeBackend.launch(args.url, headless=args.headless) as backend:
        backend.capture_screenshots = args.screenshots
        executor = DesktopExecutor(backend, policy)
        started = time.perf_counter()
        decided_on = backend.observe()  # action events carry the snapshot after the action
        for event in executor.run_iter(task):
            decision = event.decision
            label = decision.kind.value if decision.kind else decision.terminal.value
            target = decided_on.element(decision.target_id).name if decision.target_id else ""
            print(f"{time.perf_counter() - started:6.2f}s  {label:<16} {target[:40]!r}")
            decided_on = event.snapshot
            if event.result:
                print(f"\n{event.result.status.value}", event.result.reason or "")


if __name__ == "__main__":
    main()
