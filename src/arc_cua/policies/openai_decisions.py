"""ChoiceTransport for OpenAI's Decisions API."""

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Mapping, Sequence

import httpx

DEFAULT_URL = "https://api.openai.com/v1/decisions"
DEFAULT_MODEL = "gpt-6-luna"


class OpenAIDecisionsTransport:
    """Sends arc-cua's choice questions to OpenAI's Decisions API."""

    name = "OpenAI Decisions"
    supports_images = True

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 25,
        client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("Set OPENAI_API_KEY or pass api_key=...")
        self.model = model or os.environ.get("OPENAI_DECISIONS_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("OPENAI_DECISIONS_URL", DEFAULT_URL)
        self.client = client or httpx.Client(http2=True, timeout=timeout_s)

    def ask(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        images: Sequence[bytes] = (),
    ) -> Mapping[str, Any]:
        # The API needs at least two choices per question; a single option is
        # the only possible answer, so it is answered here.
        forced = {
            name: {"choice": only, "confidence": 1.0, "probabilities": {only: 1.0}}
            for name, question in questions.items()
            if len(question["criteria"]) == 1
            for only in question["criteria"]
        }
        asked = {name: question for name, question in questions.items() if name not in forced}
        if not asked:
            return {"answers": forced}
        response = self._post(_request_body(self.model, state, asked, images))
        answers, refused = _answers(response)
        return {
            "answers": {**answers, **forced},
            "refused": refused,
            "model": response.get("model"),
            "usage": response.get("usage"),
        }

    def _post(self, body: Mapping[str, Any]) -> Mapping[str, Any]:
        for attempt in range(3):
            try:
                response = self.client.post(
                    self.base_url,
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
            except httpx.HTTPError as exc:
                raise RuntimeError("OpenAI Decisions connection failed; no action executed") from exc
            if response.status_code in {429, 500, 502, 503} and attempt < 2:
                time.sleep(0.5 * (2**attempt))
                continue
            if response.is_error:
                code = _error_code(response)
                detail = f" ({code})" if code else ""
                raise RuntimeError(
                    f"OpenAI Decisions returned HTTP {response.status_code}{detail}; no action executed"
                )
            return response.json()
        raise RuntimeError("OpenAI Decisions unavailable")


# ---- wire format ------------------------------------------------------------------


def _request_body(
    model: str,
    state: Mapping[str, Any],
    questions: Mapping[str, Any],
    images: Sequence[bytes],
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": json.dumps(state, ensure_ascii=False)}]
    for image in images:
        encoded = base64.b64encode(image).decode()
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{encoded}"})
    return {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "questions": [
            {
                "type": "choice",
                "name": name,
                "instructions": json.dumps(question.get("instructions", {}), ensure_ascii=False),
                "choices": [
                    {"value": value, "description": _text(description)}
                    for value, description in question["criteria"].items()
                ],
            }
            for name, question in questions.items()
        ],
    }


def _answers(response: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    answers: dict[str, dict[str, Any]] = {}
    refused: list[str] = []
    for answer in response.get("answers", []):
        if answer.get("type") == "refusal":
            refused.append(answer.get("name"))
            continue
        if answer.get("type") != "choice":
            continue
        probabilities = answer.get("probabilities")
        answers[answer.get("name")] = {
            "choice": answer.get("choice"),
            "confidence": answer.get("confidence"),
            "probabilities": (
                {item.get("value"): item.get("probability") for item in probabilities}
                if isinstance(probabilities, list) else probabilities
            ),
        }
    return answers, refused


def _text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _error_code(response: httpx.Response) -> str | None:
    """Read only the structured error code, never echo arbitrary response text."""
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        return None
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) else None
