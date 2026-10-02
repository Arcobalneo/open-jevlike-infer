"""Minimal Python client: route a support ticket, with an optional screenshot.

pip install httpx
python examples/client.py [screenshot.png]
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys

import httpx

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")


def decide(state, questions, images=(), videos=()) -> dict:
    response = httpx.post(
        f"{BASE}/v1/systemone",
        json={
            "model": "clef-flash",
            "state": state,
            "questions": questions,
            "images": list(images),
            "videos": list(videos),
        },
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(response.json()["error"]["message"])
    return response.json()["answers"]


def image_ref(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


if __name__ == "__main__":
    images = [image_ref(path) for path in sys.argv[1:]]
    answers = decide(
        state={
            "ticket": "The export button does nothing since this morning. Screenshot attached.",
            "plan": "enterprise",
        },
        questions={
            "team": {
                "type": "choice",
                "criteria": {"billing": "Payments, invoices", "technical": "Bugs, outages", "sales": "Pricing"},
            },
            "priority": {"type": "score", "criteria": ["low", "normal", "high", "urgent"]},
            "needs_human": {"type": "noul", "instructions": "Does this need a human agent?"},
        },
        images=images,
    )
    team = answers["team"]
    print(f"team: {team['choice']} ({team['confidence']:.0%})")
    print(f"priority: {answers['priority']['score']:.2f} of 3")
    print(f"needs a human: {answers['needs_human']['noul']:.0%}")
