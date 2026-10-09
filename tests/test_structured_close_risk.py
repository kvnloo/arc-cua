"""Regressions for shhivv/arc-cua#5; diagnosis and element-based fix: rymalia."""

import pytest

from arc_cua import (
    ActionKind,
    Decision,
    DesktopElement,
    DesktopExecutor,
    DesktopSnapshot,
    Subtask,
    TerminalKind,
)
from arc_cua.backends import StateMachineBackend
from arc_cua.policies import ChoicePolicy, ScriptedPolicy
from arc_cua.runtime import RuntimeConfig

# The metadata must add to, never replace, the existing label-based risk.
CASES = [
    ("", "close", (), False),
    ("", "close", ("delete",), False),
    ("", "close", ("close",), True),
    ("Fenster schließen", "close", (), False),
    ("Fenster schließen", "close", ("close",), True),
    ("Delete document", "close", ("close",), False),
    ("Delete document", "close", ("delete",), False),
    ("Delete document", "close", ("close", "delete"), True),
    ("Close", None, (), False),
    ("", "minimize", (), True),
    ("", "fullscreen", (), True),
    ("", "unknown", (), True),
    ("", None, (), True),
]


class Recorder:
    name = "Recorder"

    def __init__(self):
        self.calls = []

    def ask(self, state, questions, *, images=()):
        self.calls.append(questions)
        return {"answers": {"operation": {
            "choice": "NEEDS_AGENT",
            "confidence": 0.9,
            "probabilities": {
                option: float(option == "NEEDS_AGENT")
                for option in questions["operation"]["criteria"]
            },
        }}}


def snapshot(state, label, control):
    metadata = {} if control is None else {"window_control": control}
    return DesktopSnapshot(
        application="Fixture", window="Untitled", revision=str(state["activations"]),
        elements=(
            DesktopElement(
                id="target", role="Button", name=label,
                actions=(ActionKind.CLICK, ActionKind.DOUBLE_CLICK), metadata=metadata,
            ),
            DesktopElement(
                id="safe", role="Button", name="Save",
                actions=(ActionKind.CLICK, ActionKind.DOUBLE_CLICK),
            ),
        ),
    )


@pytest.mark.parametrize(("label", "control", "allowed", "permitted"), CASES)
def test_policy_uses_structured_control_risks(label, control, allowed, permitted):
    transport = Recorder()
    observed = snapshot({"activations": 0}, label, control)
    metadata_before = dict(observed.elements[0].metadata)
    ChoicePolicy(transport).decide(
        subtask=Subtask(goal="Update the document", verification=("Updated",), allowed_risks=allowed),
        snapshot=observed, history=(),
    )
    for question in ("click_target", "double_click_target"):
        offered = set(transport.calls[0][question]["criteria"])
        assert "safe" in offered
        assert ("target" in offered) is permitted
    assert observed.elements[0].name == label
    assert observed.elements[0].metadata == metadata_before


@pytest.mark.parametrize(("label", "control", "allowed", "permitted"), CASES)
@pytest.mark.parametrize("kind", [ActionKind.CLICK, ActionKind.DOUBLE_CLICK])
@pytest.mark.parametrize("dry_run", [False, True])
def test_runtime_gates_custom_policy_before_dispatch(label, control, allowed, permitted, kind, dry_run):
    def observe(state):
        return snapshot(state, label, control)

    def activate(state, action):
        state["activations"] += 1

    backend = StateMachineBackend({"activations": 0}, observe, activate)
    policy = ScriptedPolicy([
        Decision(kind=kind, target_id="target"),
        Decision(terminal=TerminalKind.SUBTASK_COMPLETE),
    ])
    result = DesktopExecutor(
        backend, policy,
        config=RuntimeConfig(dry_run=dry_run, post_action_settle_s=0, late_reaction_s=0),
    ).run(Subtask(goal="Update the document", verification=("Updated",), allowed_risks=allowed))
    if not permitted:
        assert result.status == TerminalKind.NEEDS_AGENT
        assert "allowed_risks" in result.reason
        assert backend.state["activations"] == 0
    elif dry_run:
        assert result.status == TerminalKind.DRY_RUN
        assert backend.state["activations"] == 0
    else:
        assert result.status == TerminalKind.SUBTASK_COMPLETE
        assert backend.state["activations"] == 1
