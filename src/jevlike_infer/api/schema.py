"""Validation of Jev / SystemOne ``POST /v1/systemone`` request bodies.

A request has ``model``, ``state`` (any JSON value) and ``questions``: a mapping of question ID to a
question with ``type`` ``noul`` (true/false), ``choice`` (named options) or ``score`` (ordered levels).
Limits follow Jev: choice 1..255 options, score 2..10 levels, noul criteria keys true/false only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jevlike_infer.errors import RequestError

QUESTION_TYPES = ("noul", "choice", "score")
NOUL_KEYS = {"true", "false"}


@dataclass(frozen=True)
class Limits:
    max_choice_options: int = 255
    min_score_levels: int = 2
    max_score_levels: int = 10
    max_media_items: int = 16


def validate_request(body: Any, limits: Limits = Limits()) -> None:
    """Raise :class:`RequestError` with a client-facing message if ``body`` is not a valid request."""
    if not isinstance(body, dict):
        raise RequestError("request body must be a JSON object")
    if not isinstance(body.get("model"), str) or not body["model"]:
        raise RequestError("model is required and must be a string")
    if "state" not in body:
        raise RequestError("state is required")
    questions = body.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise RequestError("questions must be a non-empty object")
    for question_id, question in questions.items():
        _validate_question(question_id, question, limits)
    for key in ("images", "videos"):
        if body.get(key) is not None and not isinstance(body[key], list):
            raise RequestError(f"{key} must be a list")
    media_count = len(body.get("images") or []) + len(body.get("videos") or [])
    if media_count > limits.max_media_items:
        raise RequestError(f"at most {limits.max_media_items} images and videos per request")
    if body.get("media_kwargs") is not None and not isinstance(body["media_kwargs"], dict):
        raise RequestError("media_kwargs must be an object")


def _validate_question(question_id: str, question: Any, limits: Limits) -> None:
    if not question_id:
        raise RequestError("question IDs must be non-empty")
    if not isinstance(question, dict):
        raise RequestError(f"{question_id}: question must be an object")
    question_type = question.get("type")
    if question_type not in QUESTION_TYPES:
        raise RequestError(f"{question_id}: type must be noul, choice, or score")
    instructions = question.get("instructions")
    if instructions is not None and not isinstance(instructions, str):
        raise RequestError(f"{question_id}: instructions must be a string")
    criteria = question.get("criteria")
    if question_type == "choice":
        if not isinstance(criteria, dict) or not criteria:
            raise RequestError(f"{question_id}: choice criteria must be a non-empty object of option ID to description")
        if len(criteria) > limits.max_choice_options:
            raise RequestError(f"{question_id}: choice allows at most {limits.max_choice_options} options")
        if any(not option_id for option_id in criteria):
            raise RequestError(f"{question_id}: choice option IDs must be non-empty")
    elif question_type == "score":
        if not isinstance(criteria, list):
            raise RequestError(f"{question_id}: score criteria must be a list of level descriptions")
        if not limits.min_score_levels <= len(criteria) <= limits.max_score_levels:
            raise RequestError(
                f"{question_id}: score needs {limits.min_score_levels} to {limits.max_score_levels} levels"
            )
    elif criteria is not None and (not isinstance(criteria, dict) or not set(criteria) <= NOUL_KEYS):
        raise RequestError(f"{question_id}: noul criteria may only have keys true and false")
