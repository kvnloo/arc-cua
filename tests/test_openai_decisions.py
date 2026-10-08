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
    return httpx.Client(transport=httpx.MockTransport(responder(selections, captured, refuse=refuse)))


def responder(selections, captured, *, refuse=()):
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
    return respond


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


QUESTION = {"q": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}


def answer_a(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"answers": [{
        "type": "choice", "name": "q", "choice": "a", "confidence": 0.9,
        "probabilities": [{"value": "a", "probability": 0.95}, {"value": "b", "probability": 0.05}],
    }]})


def test_connection_errors_and_timeouts_are_retried(monkeypatch) -> None:
    monkeypatch.setattr("arc_cua.policies.openai_decisions.time.sleep", lambda _: None)
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("slow", request=request)
        if len(calls) == 2:
            return httpx.Response(504)
        return answer_a(request)

    transport = OpenAIDecisionsTransport(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    assert transport.ask({}, QUESTION)["answers"]["q"]["choice"] == "a"
    assert len(calls) == 3


def test_retry_after_is_honoured(monkeypatch) -> None:
    slept = []
    monkeypatch.setattr("arc_cua.policies.openai_decisions.time.sleep", slept.append)
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, headers={"retry-after": "1.5"}) if len(calls) == 1 else answer_a(request)

    transport = OpenAIDecisionsTransport(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    transport.ask({}, QUESTION)
    assert slept == [1.5]


def test_errors_without_a_code_report_the_parameter() -> None:
    def respond(request):
        return httpx.Response(400, json={"error": {"code": None, "param": "questions[1].name", "message": "x"}})

    transport = OpenAIDecisionsTransport(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(RuntimeError, match=r"HTTP 400 \(param questions\[1\]\.name\)"):
        transport.ask({}, QUESTION)


def test_image_detail_is_sent_when_set() -> None:
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return answer_a(request)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    OpenAIDecisionsTransport(api_key="test", client=client, image_detail="low").ask({}, QUESTION, images=(b"png",))
    assert captured[0]["input"][0]["content"][1]["detail"] == "low"


def test_warm_sends_a_minimal_decision() -> None:
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={"answers": []})

    OpenAIDecisionsTransport(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond))).warm()
    assert [len(q["choices"]) for q in captured[0]["questions"]] == [2]


def test_screenshot_check_refusal_overrides_the_text_answer() -> None:
    text_answers = responder({"operation": "SUBTASK_COMPLETE", "verification_0": "SATISFIED"}, [])

    def respond(request):
        body = json.loads(request.content)
        if body["input"][0]["content"][1:]:  # the screenshot check
            return httpx.Response(200, json={"answers": [{"type": "refusal", "name": "verification_0"}]})
        return text_answers(request)

    transport = OpenAIDecisionsTransport(api_key="test", client=httpx.Client(transport=httpx.MockTransport(respond)))
    with pytest.raises(InvalidChoiceResponse, match="refused question verification_0"):
        ChoicePolicy(transport, screenshot_checks=True, invalid_retries=0).decide(
            subtask=Subtask(goal="Review", verification=("Reviewed",)), snapshot=snapshot(lambda: b"png"), history=(),
        )
