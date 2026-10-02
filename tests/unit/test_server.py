"""HTTP behaviour with a fake model: no GPU, no weights."""

import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from jevlike_infer.api.server import create_app
from jevlike_infer.config import Settings
from jevlike_infer.errors import OverloadedError
from jevlike_infer.media import Video
from jevlike_infer.models.base import DecisionModel


class FakeModel(DecisionModel):
    def __init__(self):
        self.seen = []

    def decide(self, request):
        self.seen.append(request)
        if request["state"] == "too long":
            raise ValueError("schema requires 99999 tokens before state; maximum is 16384")
        if request["state"] == "oom":
            raise OverloadedError("GPU out of memory")
        if request["state"] == "crash":
            raise RuntimeError("boom")
        return {
            "model": request["model"],
            "answers": {q: {"type": "noul", "noul": 0.5} for q in request["questions"]},
            "usage": {"input_tokens": 3, "output_tokens": 0},
        }


@pytest.fixture
def model():
    return FakeModel()


def client(model, **settings):
    return TestClient(create_app(model, Settings(model_path="/unused", **settings)))


def post(c, body, **kwargs):
    return c.post("/v1/systemone", json=body, **kwargs)


BODY = {"model": "clef-flash", "state": "x", "questions": {"q": {"type": "noul"}}}


def png_uri():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_health_and_models(model):
    c = client(model, served_model_name="my-clef")
    assert c.get("/health").json() == {"status": "ok", "model": "my-clef"}
    assert c.get("/v1/models").json()["data"][0]["id"] == "my-clef"


def test_success(model):
    response = post(client(model), BODY)
    assert response.status_code == 200
    assert response.json()["answers"]["q"]["noul"] == 0.5


def test_media_is_decoded_before_reaching_the_model(model):
    post(client(model), {**BODY, "images": [png_uri()], "videos": [[png_uri(), png_uri()]]})
    request = model.seen[-1]
    assert isinstance(request["images"][0], Image.Image)
    assert isinstance(request["videos"][0], Video) and request["videos"][0].frames.shape[0] == 2


@pytest.mark.parametrize(
    ("body", "status", "error_type"),
    [
        ({"state": "x"}, 400, "invalid_request_error"),
        ({**BODY, "images": ["@@@"]}, 400, "invalid_request_error"),
        ({**BODY, "state": "too long"}, 400, "invalid_request_error"),
        ({**BODY, "state": "oom"}, 503, "overloaded_error"),
        ({**BODY, "state": "crash"}, 500, "api_error"),
    ],
)
def test_errors(model, body, status, error_type):
    response = post(client(model), body)
    assert response.status_code == status
    assert response.json()["error"]["type"] == error_type and response.json()["error"]["message"]


def test_invalid_json(model):
    response = client(model).post("/v1/systemone", content=b"{nope", headers={"content-type": "application/json"})
    assert response.status_code == 400 and "JSON" in response.json()["error"]["message"]


def test_api_key(model):
    c = client(model, api_key="secret")
    assert post(c, BODY).status_code == 401
    assert post(c, BODY, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert post(c, BODY, headers={"Authorization": "Bearer secret"}).status_code == 200
    assert c.get("/health").status_code == 200  # health stays open for probes
