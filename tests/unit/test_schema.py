import pytest

from jevlike_infer.api.schema import Limits, validate_request
from jevlike_infer.errors import RequestError

NOUL = {"q": {"type": "noul"}}


def body(**overrides):
    return {"model": "m", "state": "x", "questions": NOUL, **overrides}


@pytest.mark.parametrize(
    "request_body",
    [
        body(),
        body(state=None),
        body(state={"a": [1, {"b": None}]}),
        body(questions={"c": {"type": "choice", "criteria": {"a": None}}}),
        body(questions={"c": {"type": "choice", "criteria": {str(i): None for i in range(255)}}}),
        body(questions={"s": {"type": "score", "criteria": ["lo", "hi"]}}),
        body(questions={"s": {"type": "score", "criteria": [str(i) for i in range(10)]}}),
        body(questions={"n": {"type": "noul", "criteria": {"true": "yes"}}}),
        body(questions={"n": {"type": "noul", "instructions": ""}}),
        body(images=[], videos=None, media_kwargs={"images_kwargs": {}}),
    ],
)
def test_valid(request_body):
    validate_request(request_body)


@pytest.mark.parametrize(
    ("request_body", "message"),
    [
        ([1], "JSON object"),
        ({"state": "x", "questions": NOUL}, "model"),
        (body(model=1), "model"),
        ({"model": "m", "questions": NOUL}, "state"),
        (body(questions={}), "questions"),
        (body(questions=[{"type": "noul"}]), "questions"),
        (body(questions={"q": "noul"}), "object"),
        (body(questions={"q": {"type": "bool"}}), "type"),
        (body(questions={"q": {"type": "noul", "instructions": 5}}), "instructions"),
        (body(questions={"q": {"type": "choice"}}), "criteria"),
        (body(questions={"q": {"type": "choice", "criteria": ["a"]}}), "criteria"),
        (body(questions={"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}}), "255"),
        (body(questions={"q": {"type": "choice", "criteria": {"": None}}}), "non-empty"),
        (body(questions={"q": {"type": "score", "criteria": ["one"]}}), "levels"),
        (body(questions={"q": {"type": "score", "criteria": [str(i) for i in range(11)]}}), "levels"),
        (body(questions={"q": {"type": "score", "criteria": {"a": 1}}}), "list"),
        (body(questions={"q": {"type": "noul", "criteria": {"yes": "y"}}}), "true"),
        (body(images="abc"), "list"),
        (body(images=["x"] * 17), "at most 16"),
        (body(media_kwargs=[1]), "media_kwargs"),
    ],
)
def test_invalid(request_body, message):
    with pytest.raises(RequestError, match=message):
        validate_request(request_body)


def test_custom_media_limit():
    with pytest.raises(RequestError, match="at most 2"):
        validate_request(body(images=["a", "b", "c"]), Limits(max_media_items=2))
