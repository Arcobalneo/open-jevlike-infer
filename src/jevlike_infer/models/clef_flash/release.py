"""Access to the Clef-Flash release code shipped with the weights.

The model repository ships ``joint_schema_model.py`` (record encoding, the joint schema head and
``systemone``). It is imported from the downloaded snapshot so the code always matches the weights.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ID = "Cloudflare/clef-flash"
REVISION = "17f0b0ad64efb65d273590632833508766b2aae6"  # validated revision (2026-10-01)


def import_release(model_path: str | Path) -> ModuleType:
    path = Path(model_path) / "joint_schema_model.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; download the model with scripts/download_model.py --model clef-flash"
        )
    if "joint_schema_model" in sys.modules:
        return sys.modules["joint_schema_model"]
    spec = importlib.util.spec_from_file_location("joint_schema_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = module
    spec.loader.exec_module(module)
    return module


def to_release_record(request: dict[str, Any]) -> dict[str, Any]:
    """Convert a decoded request into the record format of ``joint_schema_model.encode_record``.

    Videos are already sampled by the server, so the processor gets their metadata and frame
    sampling is turned off; otherwise the Qwen3-VL processor resamples every clip again.
    """
    record = dict(request)
    videos = request.get("videos") or []
    record["videos"] = [video.frames for video in videos]
    if videos:
        record["media_kwargs"] = {
            **(request.get("media_kwargs") or {}),
            "do_sample_frames": False,
            "video_metadata": [video.metadata for video in videos],
        }
    return record
