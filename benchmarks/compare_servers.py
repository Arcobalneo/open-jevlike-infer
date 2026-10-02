"""Compare the answers of two running servers question by question (e.g. vLLM vs transformers backend).

    python benchmarks/compare_servers.py --a http://127.0.0.1:8000 --b http://127.0.0.1:8001

Reports, per question, whether the top option matches and the total-variation distance between the
two probability distributions (0 = identical, 1 = disjoint), plus whether usage.input_tokens agree.
"""

from __future__ import annotations

import argparse
import base64
import io
import sys

import httpx
from PIL import Image


def image(color: str) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (224, 224), color).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


REQUESTS = [
    {
        "model": "cmp",
        "state": "Our checkout started returning errors and orders are blocked.",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages"},
            },
            "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today"]},
            "outage": {"type": "noul", "instructions": "Is a service down?"},
        },
    },
    {
        "model": "cmp",
        "state": {"invoice": {"vendor": "Acme", "total": 1250.0, "status": "overdue"}},
        "questions": {
            "status": {"type": "choice", "criteria": {"paid": None, "overdue": None, "draft": None}},
            "large": {"type": "noul", "instructions": "Total above 1000?"},
        },
    },
    {
        "model": "cmp",
        "state": "用户说：我昨天买的手机屏幕碎了，要求退货退款。",
        "questions": {"intent": {"type": "choice", "criteria": {"退货退款": None, "咨询物流": None, "表扬": None}}},
    },
    {
        "model": "cmp",
        "state": "The quarterly report lists revenue. " * 800 + "The Berlin office was closed.",
        "questions": {"closed": {"type": "noul", "instructions": "Was the Berlin office closed?"}},
    },
    {
        "model": "cmp",
        "state": "Two pictures.",
        "images": [image("red"), image("blue")],
        "questions": {
            "second": {
                "type": "choice",
                "instructions": "Color of the second picture?",
                "criteria": {"red": None, "blue": None, "green": None},
            }
        },
    },
    {
        "model": "cmp",
        "state": "A short clip.",
        "videos": [[image("green")] * 4],
        "questions": {
            "color": {"type": "choice", "instructions": "Color of the video?", "criteria": {"red": None, "green": None}}
        },
    },
]


def distribution(answer: dict) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1 - answer["noul"]}
    return answer["probabilities"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--max-tv", type=float, default=0.02, help="fail if any question differs more than this")
    args = parser.parse_args()
    client = httpx.Client(timeout=300)
    worst, mismatches = 0.0, 0
    for body in REQUESTS:
        a = client.post(f"{args.a.rstrip('/')}/v1/systemone", json=body).json()
        b = client.post(f"{args.b.rstrip('/')}/v1/systemone", json=body).json()
        if a["usage"] != b["usage"]:
            print(f"usage differs: {a['usage']} vs {b['usage']}")
            mismatches += 1
        for qid, answer in a["answers"].items():
            p, q = distribution(answer), distribution(b["answers"][qid])
            tv = 0.5 * sum(abs(p[k] - q[k]) for k in p)
            same = max(p, key=p.get) == max(q, key=q.get)
            worst = max(worst, tv)
            mismatches += not same
            print(f"{qid:12s} top_same={same!s:5s} TV={tv:.4f}")
    print(f"\nmax TV {worst:.4f}, mismatches {mismatches}")
    return 1 if mismatches or worst > args.max_tv else 0


if __name__ == "__main__":
    sys.exit(main())
