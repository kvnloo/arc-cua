"""OpenAIDecisionsTransport against a mocked endpoint."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from arc_cua import ActionKind, DesktopElement, DesktopSnapshot, Subtask, TerminalKind
from arc_cua.policies import ChoicePolicy, InvalidChoiceResponse, OpenAIDecisionsTransport


def snapshot(screenshot=None) -> DesktopSnapshot:
    return DesktopSnapshot(application="Chrome", window="Checkout", revision="1", screenshot=screenshot, elements=(
        DesktopElement(id="w1", role="button", name="Review", actions=(ActionKind.CLICK,)),
        DesktopElement(id="w2", role="button", name="Cancel", actions=(ActionKind.CLICK,)),
    ))


def mock(selections, captured, *, refuse=()):
    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured.append(body)
        answers = []
        for question in body["questions"]:
            name = question["name"]
            if name in refuse:
                answers.append({"type": "refusal", "name": name})
                continue
            if name not in selections:
                continue
            choice = selections[name]
            others = [c["value"] for c in question["choices"] if c["value"] != choice]
            probabilities = [{"value": choice, "probability": 1 - 0.01 * len(others)}]
            probabilities += [{"value": value, "probability": 0.01} for value in others]
            answers.append({
                "type": "choice", "name": name, "choice": choice,
                "probabilities": probabilities, "confidence": 0.9,
            })
        return httpx.Response(200, json={"answers": answers})
    return httpx.Client(transport=httpx.MockTransport(respond))


def test_choice_policy_decides_through_openai() -> None:
    captured = []
    client = mock({"operation": "CLICK", "click_target": "w2", "click_modifier": "NONE"}, captured)
    transport = OpenAIDecisionsTransport(api_key="test", client=client)
    decision = ChoicePolicy(transport).decide(
        subtask=Subtask(goal="Cancel the order", verification=("Order cancelled",)), snapshot=snapshot(), history=(),
    )
    assert (decision.kind, decision.target_id, decision.confidence) == (ActionKind.CLICK, "w2", 0.9)
    assert decision.margin is not None and decision.margin > 0.9
    body = captured[0]
    assert body["model"] == "gpt-6-luna"
    state = json.loads(body["input"][0]["content"][0]["text"])
    assert state["subtask"]["goal"] == "Cancel the order"
    heads = {q["name"]: q for q in body["questions"]}
    assert {q["type"] for q in body["questions"]} == {"choice"}
    assert {c["value"] for c in heads["click_target"]["choices"]} == {"w1", "w2"}
    assert all(isinstance(c["description"], str) for c in heads["click_target"]["choices"])


def test_invented_answer_is_rejected() -> None:
    def respond(request):
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": [
            {"type": "choice", "name": q["name"], "choice": "w9",
             "probabilities": [{"value": "w9", "probability": 1.0}], "confidence": 1.0}
            for q in body["questions"]
        ]})

    transport = OpenAIDecisionsTransport(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(ValueError, match="Invalid OpenAI Decisions choice response"):
        ChoicePolicy(transport).decide(
            subtask=Subtask(goal="Review", verification=("Reviewed",)), snapshot=snapshot(), history=(),
        )


def test_refusal_of_a_used_question_executes_nothing() -> None:
    client = mock({"operation": "CLICK", "click_target": "w1", "click_modifier": "NONE"}, [], refuse={"operation"})
    with pytest.raises(InvalidChoiceResponse, match="refused question operation; no action executed"):
        ChoicePolicy(OpenAIDecisionsTransport(api_key="test", client=client)).decide(
            subtask=Subtask(goal="Review", verification=("Reviewed",)), snapshot=snapshot(), history=(),
        )


def test_refusal_of_an_unused_question_is_ignored() -> None:
    client = mock(
        {"operation": "CLICK", "click_target": "w1", "click_modifier": "NONE"}, [], refuse={"hotkey_value"},
    )
    decision = ChoicePolicy(OpenAIDecisionsTransport(api_key="test", client=client)).decide(
        subtask=Subtask(goal="Review", verification=("Reviewed",)), snapshot=snapshot(), history=(),
    )
    assert (decision.kind, decision.target_id) == (ActionKind.CLICK, "w1")
    assert decision.raw["refused"] == ["hotkey_value"]


def test_screenshot_checks_send_the_snapshot_png() -> None:
    captured = []
    client = mock({"operation": "SUBTASK_COMPLETE", "verification_0": "SATISFIED"}, captured)
    decision = ChoicePolicy(OpenAIDecisionsTransport(api_key="test", client=client), screenshot_checks=True).decide(
        subtask=Subtask(goal="Review", verification=("Reviewed",)), snapshot=snapshot(lambda: b"png-bytes"), history=(),
    )
    assert decision.terminal == TerminalKind.SUBTASK_COMPLETE
    image = captured[1]["input"][0]["content"][1]
    encoded = base64.b64encode(b"png-bytes").decode()
    assert image == {"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"}
    assert [q["name"] for q in captured[1]["questions"]] == ["verification_0"]


def test_single_option_questions_are_answered_locally() -> None:
    captured = []
    client = mock({"many": "b"}, captured)
    result = OpenAIDecisionsTransport(api_key="test", client=client).ask({}, {
        "many": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        "one": {"type": "choice", "criteria": {"only": "Only"}},
    })
    assert [q["name"] for q in captured[0]["questions"]] == ["many"]
    assert result["answers"]["one"] == {"choice": "only", "confidence": 1.0, "probabilities": {"only": 1.0}}
    assert result["answers"]["many"]["choice"] == "b"


def test_errors_report_only_the_structured_code() -> None:
    def respond(request):
        return httpx.Response(400, json={"error": {"code": "invalid_request_error", "message": "secret page text"}})

    transport = OpenAIDecisionsTransport(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(RuntimeError, match=r"HTTP 400 \(invalid_request_error\); no action executed") as error:
        transport.ask({}, {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}})
    assert "secret page text" not in str(error.value)
