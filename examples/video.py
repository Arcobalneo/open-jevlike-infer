"""Send a video file, or a list of frames, and ask about motion.

python examples/video.py clip.mp4
python examples/video.py frame1.jpg frame2.jpg frame3.jpg ...   # frames, in order
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys

import httpx

BASE = os.environ.get("BASE", "http://127.0.0.1:8000")


def ref(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        sys.exit(__doc__)
    video = ref(paths[0]) if len(paths) == 1 else [ref(p) for p in paths]
    response = httpx.post(
        f"{BASE}/v1/systemone",
        json={
            "model": "clef-flash",
            "state": "A short clip is attached.",
            "videos": [video],
            "questions": {
                "moving": {"type": "noul", "instructions": "Is something moving in the clip?"},
                "scene": {
                    "type": "choice",
                    "criteria": {"indoor": None, "outdoor": None, "synthetic": "computer-generated"},
                },
            },
        },
        timeout=120,
    )
    print(response.json())
