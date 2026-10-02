"""End-to-end checks against a running server (any model, any backend).

    python tests/e2e/run_e2e.py --base-url http://127.0.0.1:8000 [--output results.json]

Checks the response contract on every answer, request limits, error handling, long inputs, a few
semantic sanity cases, images and videos, and correctness under concurrency. Semantic expectations
are easy cases any competent decision model gets right. Exit code is non-zero if any check fails.
Performance numbers live in benchmarks/, not here.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import http.server
import io
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

import av
import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = ""
FIXTURE_HOST = "127.0.0.1"
client = httpx.Client(timeout=300)
results: list[dict] = []


# ---------------------------------------------------------------- helpers


def post(body, raw: bytes | None = None) -> httpx.Response:
    if raw is not None:
        return client.post(f"{BASE}/v1/systemone", content=raw, headers={"content-type": "application/json"})
    return client.post(f"{BASE}/v1/systemone", json=body)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""), flush=True)
    return ok


def ask(state, questions, **extra) -> dict:
    response = post({"model": "m", "state": state, "questions": questions, **extra})
    response.raise_for_status()
    return response.json()


def close(a: float, b: float, tol: float = 2e-3) -> bool:
    return abs(a - b) <= tol


def distribution(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


def total_variation(a: dict, b: dict) -> float:
    worst = 0.0
    for qid, answer in a["answers"].items():
        p, q = distribution(answer), distribution(b["answers"][qid])
        worst = max(worst, 0.5 * sum(abs(p[k] - q[k]) for k in p))
    return worst


# ---------------------------------------------------------------- response contract


def choice_problem(question: dict, answer: dict) -> str:
    options = [str(key) for key in question["criteria"]]
    if set(answer) != {"type", "choice", "confidence", "probabilities"}:
        return f"keys {sorted(answer)}"
    probs = answer["probabilities"]
    if answer["choice"] not in options or list(probs) != options:
        return "choice / probability keys do not match the criteria"
    if not close(sum(probs.values()), 1.0, 5e-3):
        return f"probabilities sum to {sum(probs.values())}"
    if (
        not close(answer["confidence"], probs[answer["choice"]], 1e-4)
        or probs[answer["choice"]] < max(probs.values()) - 1e-4
    ):
        return "choice is not the argmax or confidence is not its probability"
    return ""


def score_problem(question: dict, answer: dict) -> str:
    levels = [str(i) for i in range(len(question["criteria"]))]
    if set(answer) != {"type", "score", "confidence", "legend", "probabilities"}:
        return f"keys {sorted(answer)}"
    probs = answer["probabilities"]
    if list(probs) != levels or answer["legend"] != dict(zip(levels, question["criteria"], strict=True)):
        return "legend / probability keys wrong"
    if not close(sum(probs.values()), 1.0, 5e-3):
        return f"probabilities sum to {sum(probs.values())}"
    expected = sum(i * probs[level] for i, level in enumerate(levels))
    if not close(answer["score"], expected) or not 0 <= answer["score"] <= len(levels) - 1:
        return f"score {answer['score']} is not the expected level {expected:.4f}"
    if not close(answer["confidence"], max(probs.values()), 1e-4):
        return "confidence is not the max probability"
    return ""


def response_problem(body: dict, response: dict) -> str:
    if set(response) != {"model", "answers", "usage"}:
        return f"top-level keys {sorted(response)}"
    if response["model"] != body["model"]:
        return "model not echoed"
    if list(response["answers"]) != list(body["questions"]):
        return "answer keys / order differ from questions"
    usage = response["usage"]
    if usage.get("output_tokens") != 0 or not usage.get("input_tokens", 0) > 0:
        return f"usage {usage}"
    for qid, question in body["questions"].items():
        answer = response["answers"][qid]
        if question["type"] == "choice":
            problem = choice_problem(question, answer)
        elif question["type"] == "score":
            problem = score_problem(question, answer)
        else:
            ok = set(answer) == {"type", "noul"} and 0 <= answer["noul"] <= 1
            problem = "" if ok else f"noul answer {answer}"
        if problem:
            return f"{qid}: {problem}"
    return ""


def run(name: str, body: dict, expect=None) -> dict | None:
    """POST a valid request, check the contract, then an optional expectation(answers) -> (ok, detail)."""
    started = time.perf_counter()
    response = post(body)
    ms = (time.perf_counter() - started) * 1000
    if response.status_code != 200:
        check(name, False, f"HTTP {response.status_code}: {response.text[:300]}")
        return None
    data = response.json()
    problem = response_problem(body, data)
    if problem:
        check(name, False, f"contract: {problem}")
        return data
    ok, detail = expect(data["answers"]) if expect else (True, "")
    check(name, ok, f"{detail} ({ms:.0f} ms, {data['usage']['input_tokens']} tokens)".strip())
    return data


def expect_error(name: str, body=None, raw: bytes | None = None, status: int = 400, contains: str = "") -> None:
    response = post(body, raw)
    try:
        error = response.json().get("error", {})
    except Exception:
        error = {}
    message = error.get("message", "")
    ok = response.status_code == status and bool(message) and contains.lower() in message.lower()
    check(name, ok, f"HTTP {response.status_code} {message or response.text[:200]!r}")


# ---------------------------------------------------------------- media fixtures


def image_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def data_uri(image: Image.Image, fmt: str = "PNG") -> str:
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(image_bytes(image, fmt)).decode()


def solid(color, size=(224, 224)) -> Image.Image:
    return Image.new("RGB", size, color)


def shape(kind: str) -> Image.Image:
    image = solid("white", (256, 256))
    draw = ImageDraw.Draw(image)
    if kind == "circle":
        draw.ellipse((48, 48, 208, 208), fill="black")
    elif kind == "square":
        draw.rectangle((48, 48, 208, 208), fill="black")
    else:
        draw.polygon([(128, 40), (216, 216), (40, 216)], fill="black")
    return image


def text_image(text: str) -> Image.Image:
    image = solid("white", (640, 200))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
    except OSError:
        font = ImageFont.load_default(size=56)
    ImageDraw.Draw(image).text((30, 60), text, fill="black", font=font)
    return image


def moving_ball(direction: str, count: int = 8) -> list[np.ndarray]:
    frames = []
    for i in range(count):
        image = solid("white", (320, 160))
        x = 20 + i * 280 // (count - 1) if direction == "right" else 300 - i * 280 // (count - 1)
        ImageDraw.Draw(image).ellipse((x - 18, 62, x + 18, 98), fill="red")
        frames.append(np.asarray(image))
    return frames


def mp4_bytes(frames: list[np.ndarray], fps: int = 4) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        for array in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(array, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def serve_files(directory: Path) -> tuple[str, http.server.ThreadingHTTPServer]:
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, *args):
            pass

    bind = "127.0.0.1" if FIXTURE_HOST == "127.0.0.1" else "0.0.0.0"
    server = http.server.ThreadingHTTPServer((bind, 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://{FIXTURE_HOST}:{server.server_address[1]}", server


MOTION = {
    "dir": {
        "type": "choice",
        "instructions": "In which direction does the red ball move over time?",
        "criteria": {"left_to_right": "moves from left to right", "right_to_left": "moves from right to left"},
    }
}
COLOR = {
    "color": {
        "type": "choice",
        "instructions": "What is the dominant color of the image?",
        "criteria": {"red": None, "green": None, "blue": None, "yellow": None},
    }
}


# ---------------------------------------------------------------- sections


def section_contract() -> None:
    print("\n== contract ==")
    run(
        "model card example (choice + score + noul, score without instructions)",
        {
            "model": "clef-flash",
            "state": "Our checkout started returning errors and orders are blocked.",
            "questions": {
                "department": {
                    "type": "choice",
                    "instructions": "Which team should handle the message?",
                    "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
                },
                "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
                "outage": {"type": "noul", "instructions": "Is a service down?"},
            },
        },
        lambda a: (
            a["department"]["choice"] == "technical" and a["urgency"]["score"] > 1.0 and a["outage"]["noul"] > 0.5,
            f"department={a['department']['choice']} urgency={a['urgency']['score']} outage={a['outage']['noul']}",
        ),
    )
    run(
        "Jev-style null option descriptions",
        {
            "model": "jev-latest",
            "state": "A customer writes: my card was charged twice.",
            "questions": {
                "urgent": {"type": "noul", "instructions": "This needs a reply today."},
                "team": {
                    "type": "choice",
                    "instructions": "Which team?",
                    "criteria": {"billing": None, "shipping": None},
                },
                "tone": {
                    "type": "score",
                    "instructions": "How upset is the customer?",
                    "criteria": ["calm", "annoyed", "angry"],
                },
            },
        },
        lambda a: (a["team"]["choice"] == "billing", f"team={a['team']['choice']}"),
    )
    run(
        "JSON object state",
        {
            "model": "clef-flash",
            "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "currency": "USD", "status": "overdue"}},
            "questions": {
                "status": {
                    "type": "choice",
                    "instructions": "What is the invoice status?",
                    "criteria": {"paid": "Invoice is paid.", "overdue": "Invoice is past due.", "draft": "Not sent."},
                },
                "large": {"type": "noul", "instructions": "Is the total above 1000 USD?"},
            },
        },
        lambda a: (a["status"]["choice"] == "overdue" and a["large"]["noul"] > 0.5, f"status={a['status']['choice']}"),
    )
    states = [
        ("string", "The sky is green."),
        ("number", 42),
        ("array", [1, 2, 3]),
        ("null", None),
        ("empty string", ""),
        ("boolean", True),
        ("nested JSON", {"a": {"b": [1, {"c": None}]}}),
    ]
    for label, state in states:
        run(
            f"state type: {label}",
            {"model": "m", "state": state, "questions": {"q": {"type": "noul", "instructions": "Is the state empty?"}}},
        )
    run(
        "arbitrary model name is echoed",
        {"model": "my-custom-name/1.0", "state": "x", "questions": {"q": {"type": "noul"}}},
    )
    run(
        "instructions omitted (question ID is used)",
        {
            "model": "m",
            "state": "I love this product, five stars!",
            "questions": {
                "is_positive_review": {"type": "noul"},
                "sentiment": {"type": "choice", "criteria": {"positive": None, "negative": None, "neutral": None}},
            },
        },
        lambda a: (
            a["is_positive_review"]["noul"] > 0.5 and a["sentiment"]["choice"] == "positive",
            f"sentiment={a['sentiment']['choice']}",
        ),
    )
    run("empty instructions", {"model": "m", "state": "x", "questions": {"q": {"type": "noul", "instructions": ""}}})
    run(
        "noul with custom true/false descriptions",
        {
            "model": "m",
            "state": "The server returned HTTP 503 for every request in the last 10 minutes.",
            "questions": {
                "healthy": {
                    "type": "noul",
                    "instructions": "Service health",
                    "criteria": {"true": "The service is healthy", "false": "The service is failing"},
                }
            },
        },
        lambda a: (a["healthy"]["noul"] < 0.5, f"noul={a['healthy']['noul']}"),
    )
    run(
        "noul with only a true description",
        {
            "model": "m",
            "state": "It is raining.",
            "questions": {"q": {"type": "noul", "instructions": "Weather", "criteria": {"true": "It is raining"}}},
        },
        lambda a: (a["q"]["noul"] > 0.5, f"noul={a['q']['noul']}"),
    )


def section_limits() -> None:
    print("\n== limits and encodings ==")
    run(
        "choice with 1 option has confidence 1",
        {"model": "m", "state": "x", "questions": {"q": {"type": "choice", "criteria": {"only": None}}}},
        lambda a: (close(a["q"]["confidence"], 1.0), f"confidence={a['q']['confidence']}"),
    )
    many = {f"opt{i:03d}": f"Option number {i}" for i in range(255)}
    run(
        "choice with 255 options (Jev maximum)",
        {"model": "m", "state": "Pick option number 137.", "questions": {"q": {"type": "choice", "criteria": many}}},
    )
    run(
        "score with 2 levels (minimum)",
        {
            "model": "m",
            "state": "Everything is on fire.",
            "questions": {"q": {"type": "score", "instructions": "Severity", "criteria": ["low", "high"]}},
        },
        lambda a: (a["q"]["score"] > 0.5, f"score={a['q']['score']}"),
    )
    run(
        "score with 10 levels (maximum)",
        {
            "model": "m",
            "state": "Rate the number seven.",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "Which number from 0 to 9 is mentioned?",
                    "criteria": [str(i) for i in range(10)],
                }
            },
        },
        lambda a: (abs(a["q"]["score"] - 7) < 1.5, f"score={a['q']['score']}"),
    )
    run(
        "score levels described by objects",
        {
            "model": "m",
            "state": "Minor typo on the docs page.",
            "questions": {
                "q": {
                    "type": "score",
                    "instructions": "Severity",
                    "criteria": [{"label": "minor"}, {"label": "major"}, {"label": "critical"}],
                }
            },
        },
        lambda a: (a["q"]["score"] < 1.0, f"score={a['q']['score']}"),
    )
    questions = {f"q{i}": {"type": "noul", "instructions": f"Is {i} an even number?"} for i in range(30)}
    data = run("30 questions in one request", {"model": "m", "state": "Answer about integers.", "questions": questions})
    if data:
        correct = sum((data["answers"][f"q{i}"]["noul"] > 0.5) == (i % 2 == 0) for i in range(30))
        check("30 parity questions at least 80% correct", correct >= 24, f"{correct}/30")
    run(
        "Chinese state, option IDs and instructions",
        {
            "model": "m",
            "state": "用户说：我昨天买的手机屏幕碎了，要求退货退款。",
            "questions": {
                "意图": {
                    "type": "choice",
                    "instructions": "用户的意图是什么？",
                    "criteria": {"退货退款": "要求退货或退款", "咨询物流": "询问快递进度", "表扬": "夸奖产品"},
                },
                "负面情绪": {"type": "noul", "instructions": "用户是否不满？"},
            },
        },
        lambda a: (a["意图"]["choice"] == "退货退款", f"intent={a['意图']['choice']}"),
    )
    run(
        "emoji, quotes, markup and newlines",
        {
            "model": "m",
            "state": '🔥🔥 prod down!!!\n\t"quote" <tag> & \' \\ ',
            "questions": {"q": {"type": "noul", "instructions": "Is this an incident report?"}},
        },
        lambda a: (a["q"]["noul"] > 0.5, f"noul={a['q']['noul']}"),
    )
    choice = {"z_last": "Hardware fault", "a_first": "Software bug", "m_mid": "User error"}
    run(
        "probabilities keep the request's option order",
        {
            "model": "m",
            "state": "The app crashes with a null pointer exception after the latest release.",
            "questions": {"cause": {"type": "choice", "criteria": choice}},
        },
        lambda a: (
            list(a["cause"]["probabilities"]) == list(choice) and a["cause"]["choice"] == "a_first",
            f"order={list(a['cause']['probabilities'])}",
        ),
    )


def section_errors() -> None:
    print("\n== errors (400 with {error: {type, message}}, never 500) ==")
    q = {"q": {"type": "noul"}}
    cases = [
        ("missing model", {"state": "x", "questions": q}, "model"),
        ("model is not a string", {"model": 1, "state": "x", "questions": q}, "model"),
        ("missing state", {"model": "m", "questions": q}, "state"),
        ("missing questions", {"model": "m", "state": "x"}, "questions"),
        ("empty questions", {"model": "m", "state": "x", "questions": {}}, "questions"),
        ("questions is an array", {"model": "m", "state": "x", "questions": [{"type": "noul"}]}, "questions"),
        ("question is not an object", {"model": "m", "state": "x", "questions": {"q": "noul"}}, "object"),
        ("unknown type", {"model": "m", "state": "x", "questions": {"q": {"type": "bool"}}}, "type"),
        ("choice without criteria", {"model": "m", "state": "x", "questions": {"q": {"type": "choice"}}}, "criteria"),
        (
            "choice criteria is an array",
            {"model": "m", "state": "x", "questions": {"q": {"type": "choice", "criteria": ["a", "b"]}}},
            "criteria",
        ),
        (
            "choice with 256 options",
            {
                "model": "m",
                "state": "x",
                "questions": {"q": {"type": "choice", "criteria": {str(i): None for i in range(256)}}},
            },
            "255",
        ),
        (
            "score with 1 level",
            {"model": "m", "state": "x", "questions": {"q": {"type": "score", "criteria": ["only"]}}},
            "levels",
        ),
        (
            "score with 11 levels",
            {
                "model": "m",
                "state": "x",
                "questions": {"q": {"type": "score", "criteria": [str(i) for i in range(11)]}},
            },
            "levels",
        ),
        (
            "score criteria is an object",
            {"model": "m", "state": "x", "questions": {"q": {"type": "score", "criteria": {"a": 1, "b": 2}}}},
            "list",
        ),
        (
            "noul criteria with other keys",
            {"model": "m", "state": "x", "questions": {"q": {"type": "noul", "criteria": {"yes": "y"}}}},
            "true",
        ),
        (
            "instructions is not a string",
            {"model": "m", "state": "x", "questions": {"q": {"type": "noul", "instructions": 5}}},
            "instructions",
        ),
        ("body is an array", [1, 2], "object"),
        ("images is not a list", {"model": "m", "state": "x", "questions": q, "images": "abc"}, "list"),
        ("image is not base64", {"model": "m", "state": "x", "questions": q, "images": ["!!!notbase64!!!"]}, "base64"),
        (
            "image is base64 but not an image",
            {"model": "m", "state": "x", "questions": q, "images": [base64.b64encode(b"hello world").decode()]},
            "image",
        ),
        (
            "image URL returns 404",
            {"model": "m", "state": "x", "questions": q, "images": [f"{BASE}/no-such-image.png"]},
            "fetch",
        ),
        ("empty video frame list", {"model": "m", "state": "x", "questions": q, "videos": [[]]}, "empty"),
        (
            "video frames of different sizes",
            {
                "model": "m",
                "state": "x",
                "questions": q,
                "videos": [[data_uri(solid("red", (64, 64))), data_uri(solid("red", (32, 32)))]],
            },
            "same size",
        ),
        (
            "too many media items",
            {"model": "m", "state": "x", "questions": q, "images": [data_uri(solid("red", (8, 8)))] * 17},
            "at most",
        ),
    ]
    for name, body, contains in cases:
        expect_error(name, body, contains=contains)
    expect_error("invalid JSON", raw=b"{not json", contains="JSON")
    huge = {
        f"q{i}": {
            "type": "choice",
            "instructions": "word " * 50,
            "criteria": {f"o{j}": "long description " * 20 for j in range(40)},
        }
        for i in range(20)
    }
    expect_error(
        "schema longer than the context window", {"model": "m", "state": "x", "questions": huge}, contains="tokens"
    )
    check("server healthy after errors", client.get(f"{BASE}/health").status_code == 200)


def section_long_inputs() -> None:
    print("\n== long inputs ==")
    sentence = "The quarterly report lists revenue, costs, and headcount for each region. "
    needle = "IMPORTANT: the Berlin office was closed permanently on March 3. "
    run(
        "needle in a ~12k-token state",
        {
            "model": "m",
            "state": sentence * 400 + needle + sentence * 400,
            "questions": {"closed": {"type": "noul", "instructions": "Was the Berlin office closed?"}},
        },
        lambda a: (a["closed"]["noul"] > 0.5, f"noul={a['closed']['noul']}"),
    )
    data = run(
        "~40k-token state is truncated, not rejected",
        {
            "model": "m",
            "state": sentence * 2500,
            "questions": {"q": {"type": "noul", "instructions": "Is this a report?"}},
        },
    )
    if data:
        check(
            "truncated input stays within 16384 tokens",
            data["usage"]["input_tokens"] <= 16384,
            f"input_tokens={data['usage']['input_tokens']}",
        )


def section_semantics() -> None:
    print("\n== semantics ==")
    teams = {
        "billing": "Payments, refunds, invoices",
        "technical": "Bugs, crashes, outages",
        "sales": "Pricing questions from prospects",
    }
    nli = {"entailment": None, "neutral": None, "contradiction": None}
    cases = [
        ("I was charged twice for my subscription this month.", teams, "billing"),
        ("The mobile app crashes every time I open the settings page.", teams, "technical"),
        ("We are a 500-seat company evaluating your enterprise plan, what discounts do you offer?", teams, "sales"),
        ("A man is playing a guitar on stage. Hypothesis: A person is performing music.", nli, "entailment"),
        ("A woman is sleeping in her bed. Hypothesis: The woman is running a marathon.", nli, "contradiction"),
        (
            "What is the capital of Australia?",
            {"Sydney": None, "Canberra": None, "Melbourne": None, "Perth": None},
            "Canberra",
        ),
        (
            "Which gas do plants absorb for photosynthesis?",
            {"oxygen": None, "nitrogen": None, "carbon dioxide": None, "helium": None},
            "carbon dioxide",
        ),
        (
            "'Set an alarm for 7am tomorrow'",
            {"alarm_set": None, "weather_query": None, "play_music": None, "send_message": None},
            "alarm_set",
        ),
    ]
    correct = sum(
        ask(state, {"q": {"type": "choice", "instructions": "Pick the correct label.", "criteria": criteria}})[
            "answers"
        ]["q"]["choice"]
        == expected
        for state, criteria, expected in cases
    )
    check("choice sanity cases at least 7/8", correct >= 7, f"{correct}/{len(cases)}")
    facts = [
        ("2 + 2 = 4", True),
        ("2 + 2 = 5", False),
        ("Water boils at 100 °C at sea level.", True),
        ("The Moon is larger than the Earth.", False),
        ("Paris is in France.", True),
        ("Penguins can fly long distances.", False),
    ]
    correct = sum(
        (ask(s, {"q": {"type": "noul", "instructions": "Is the statement true?"}})["answers"]["q"]["noul"] > 0.5) == t
        for s, t in facts
    )
    check("noul facts 6/6", correct == 6, f"{correct}/6")
    levels = ["not urgent", "low", "medium", "high", "critical"]
    texts = [
        "Could you update the footer copyright year when you get a chance?",
        "A few users report the export button is slow.",
        "Login fails for about 10% of users since this morning.",
        "Production database is down, all customers see errors, revenue is stopped.",
    ]
    scores = [
        ask(t, {"u": {"type": "score", "instructions": "How urgent is this ticket?", "criteria": levels}})["answers"][
            "u"
        ]["score"]
        for t in texts
    ]
    check("score rises with severity", all(a < b for a, b in zip(scores, scores[1:], strict=False)), f"scores={scores}")

    body = {
        "model": "m",
        "state": "Refund request for a damaged item delivered yesterday.",
        "questions": {
            "a": {"type": "choice", "criteria": {"refund": None, "exchange": None, "complaint": None}},
            "b": {"type": "score", "criteria": ["calm", "upset", "furious"]},
            "c": {"type": "noul", "instructions": "Is the item damaged?"},
        },
    }
    first, second = post(body).json(), post(body).json()
    check("same request twice gives identical answers", first == second)
    shuffled = post(dict(body, questions=dict(reversed(list(body["questions"].items()))))).json()
    drift = total_variation(first, shuffled)
    same = first["answers"]["a"]["choice"] == shuffled["answers"]["a"]["choice"]
    check(
        "reordering questions keeps the decision (questions are scored jointly)",
        same and drift < 0.15,
        f"max_TV={drift:.4f}",
    )
    p1 = ask(
        body["state"], {"a": {"type": "choice", "criteria": {"refund": None, "exchange": None, "complaint": None}}}
    )["answers"]["a"]["probabilities"]
    p2 = ask(
        body["state"], {"a": {"type": "choice", "criteria": {"complaint": None, "refund": None, "exchange": None}}}
    )["answers"]["a"]["probabilities"]
    check(
        "reordering options leaves probabilities unchanged", all(close(p1[k], p2[k], 1e-3) for k in p1), f"{p1} vs {p2}"
    )


def section_media() -> None:
    print("\n== images and videos ==")
    for color in ["red", "green", "blue", "yellow"]:
        run(
            f"image data URI, solid {color}",
            {"model": "m", "state": "Look at the image.", "questions": COLOR, "images": [data_uri(solid(color))]},
            lambda a, c=color: (a["color"]["choice"] == c, f"choice={a['color']['choice']}"),
        )
    run(
        "image as raw base64 JPEG",
        {
            "model": "m",
            "state": "Look at the image.",
            "questions": COLOR,
            "images": [base64.b64encode(image_bytes(solid("blue"), "JPEG")).decode()],
        },
        lambda a: (a["color"]["choice"] == "blue", f"choice={a['color']['choice']}"),
    )
    shape_q = {
        "shape": {
            "type": "choice",
            "instructions": "Which shape is drawn?",
            "criteria": {"circle": None, "square": None, "triangle": None},
        }
    }
    for kind in ["circle", "square", "triangle"]:
        run(
            f"image shape: {kind}",
            {
                "model": "m",
                "state": "Identify the shape.",
                "questions": shape_q,
                "images": [{"url": data_uri(shape(kind))}],
            },
            lambda a, k=kind: (a["shape"]["choice"] == k, f"choice={a['shape']['choice']}"),
        )
    run(
        "receipt total read from the image",
        {
            "model": "m",
            "state": {"task": "Review the attached receipt."},
            "images": [data_uri(text_image("TOTAL: $1,250.00"))],
            "questions": {"over": {"type": "noul", "instructions": "Is the total above 1000 dollars?"}},
        },
        lambda a: (a["over"]["noul"] > 0.5, f"noul={a['over']['noul']}"),
    )
    run(
        "second of two images",
        {
            "model": "m",
            "state": "Two pictures are attached.",
            "images": [data_uri(solid("red")), data_uri(solid("blue"))],
            "questions": {
                "second": {
                    "type": "choice",
                    "instructions": "What color is the second picture?",
                    "criteria": {"red": None, "blue": None, "green": None},
                }
            },
        },
        lambda a: (a["second"]["choice"] == "blue", f"choice={a['second']['choice']}"),
    )
    run(
        "3000x2000 image",
        {"model": "m", "state": "Look.", "questions": COLOR, "images": [data_uri(solid("red", (3000, 2000)), "JPEG")]},
        lambda a: (a["color"]["choice"] == "red", f"choice={a['color']['choice']}"),
    )
    run("8x8 image", {"model": "m", "state": "Look.", "questions": COLOR, "images": [data_uri(solid("green", (8, 8)))]})
    run(
        "RGBA PNG",
        {
            "model": "m",
            "state": "Look.",
            "questions": COLOR,
            "images": [data_uri(Image.new("RGBA", (128, 128), (255, 0, 0, 255)))],
        },
        lambda a: (a["color"]["choice"] == "red", f"choice={a['color']['choice']}"),
    )
    run(
        "media_kwargs reach the processor (max_pixels)",
        {
            "model": "m",
            "state": "Look.",
            "questions": COLOR,
            "images": [data_uri(solid("red", (1024, 1024)))],
            "media_kwargs": {"images_kwargs": {"max_pixels": 128 * 128}},
        },
        lambda a: (a["color"]["choice"] == "red", f"choice={a['color']['choice']}"),
    )
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "yellow.png").write_bytes(image_bytes(solid("yellow")))
        (Path(tmp) / "right.mp4").write_bytes(mp4_bytes(moving_ball("right")))
        base, server = serve_files(Path(tmp))
        try:
            run(
                "image http URL",
                {"model": "m", "state": "Look.", "questions": COLOR, "images": [f"{base}/yellow.png"]},
                lambda a: (a["color"]["choice"] == "yellow", f"choice={a['color']['choice']}"),
            )
            run(
                "video http URL (mp4, ball moves right)",
                {"model": "m", "state": "Watch the video.", "questions": MOTION, "videos": [f"{base}/right.mp4"]},
                lambda a: (a["dir"]["choice"] == "left_to_right", f"choice={a['dir']['choice']}"),
            )
        finally:
            server.shutdown()
    run(
        "video data URI (mp4, ball moves left)",
        {
            "model": "m",
            "state": "Watch the video.",
            "questions": MOTION,
            "videos": ["data:video/mp4;base64," + base64.b64encode(mp4_bytes(moving_ball("left"))).decode()],
        },
        lambda a: (a["dir"]["choice"] == "right_to_left", f"choice={a['dir']['choice']}"),
    )
    sixteen = [data_uri(Image.fromarray(frame)) for frame in moving_ball("right", 16)]
    data = run(
        "video as a 16-frame list",
        {"model": "m", "state": "Watch the video.", "questions": MOTION, "videos": [sixteen]},
        lambda a: (a["dir"]["choice"] == "left_to_right", f"choice={a['dir']['choice']}"),
    )
    if data:
        # Clef-Flash: 16 frames -> 8 temporal groups ~ 600 tokens; resampled to 2 frames it would be ~200.
        check(
            "all 16 frames reach the model",
            data["usage"]["input_tokens"] > 450,
            f"input_tokens={data['usage']['input_tokens']}",
        )
    frames = [data_uri(solid("blue", (160, 160))) for _ in range(4)]
    run(
        "video as a 4-frame list",
        {"model": "m", "state": "Watch.", "questions": COLOR, "videos": [frames]},
        lambda a: (a["color"]["choice"] == "blue", f"choice={a['color']['choice']}"),
    )
    run(
        "image and video in one request",
        {
            "model": "m",
            "state": "An image and a video are attached.",
            "images": [data_uri(solid("red"))],
            "videos": [frames],
            "questions": {
                "img": {
                    "type": "choice",
                    "instructions": "Color of the still image?",
                    "criteria": {"red": None, "blue": None},
                },
                "vid": {
                    "type": "choice",
                    "instructions": "Color of the video?",
                    "criteria": {"red": None, "blue": None},
                },
            },
        },
        lambda a: (
            a["img"]["choice"] == "red" and a["vid"]["choice"] == "blue",
            f"img={a['img']['choice']} vid={a['vid']['choice']}",
        ),
    )


def section_concurrency() -> None:
    print("\n== concurrency ==")
    body = {
        "model": "m",
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments", "technical": "Bugs"},
            },
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    }
    reference = post(body).json()
    with cf.ThreadPoolExecutor(16) as pool:
        outcomes = list(pool.map(lambda _: post(body), range(64)))
    ok = all(r.status_code == 200 for r in outcomes)
    drift = max(total_variation(reference, r.json()) for r in outcomes) if ok else 1.0
    # Batched and single-request numerics differ in the last digits; decisions must not.
    check(
        "64 requests at concurrency 16 succeed and agree with a single request",
        ok and drift < 0.02,
        f"max_TV={drift:.4f}",
    )
    bad = {"model": "m", "state": "x", "questions": {"q": {"type": "nope"}}}
    with cf.ThreadPoolExecutor(16) as pool:
        codes = list(pool.map(lambda i: post(body if i % 2 else bad).status_code, range(40)))
    check(
        "valid and invalid requests interleaved",
        codes.count(200) == 20 and codes.count(400) == 20,
        f"200={codes.count(200)} 400={codes.count(400)}",
    )
    mixed = [
        body,
        {**body, "images": [data_uri(solid("red"))]},
        {**body, "videos": [[data_uri(solid("blue", (64, 64)))] * 4]},
        {**body, "state": "word " * 3000},
    ] * 4
    with cf.ThreadPoolExecutor(16) as pool:
        responses = list(pool.map(post, mixed))
    check(
        "text, image, video and long requests batched together",
        all(r.status_code == 200 for r in responses),
        f"statuses={sorted({r.status_code for r in responses})}",
    )


def main() -> int:
    global BASE, FIXTURE_HOST
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", help="write per-check results as JSON")
    parser.add_argument(
        "--fixture-host",
        default="127.0.0.1",
        help="address the server uses to fetch test media from this script; set it to the host IP when "
        "the server runs in a container (e.g. 172.17.0.1)",
    )
    args = parser.parse_args()
    BASE = args.base_url.rstrip("/")
    FIXTURE_HOST = args.fixture_host

    check("GET /health", client.get(f"{BASE}/health").json().get("status") == "ok")
    check("GET /v1/models", bool(client.get(f"{BASE}/v1/models").json()["data"]))
    for section in (
        section_contract,
        section_limits,
        section_errors,
        section_long_inputs,
        section_semantics,
        section_media,
        section_concurrency,
    ):
        try:
            section()
        except Exception as exc:
            check(f"{section.__name__} crashed", False, f"{type(exc).__name__}: {exc}")
    failed = [r for r in results if not r["ok"]]
    print(f"\nSUMMARY: {len(results) - len(failed)}/{len(results)} passed")
    for r in failed:
        print(f"  FAIL {r['name']} - {r['detail']}")
    if args.output:
        Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
