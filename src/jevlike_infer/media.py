"""Decoding of request images and videos.

Over HTTP a media item is a reference: ``data:<mime>;base64,...``, raw base64, an ``http(s)://`` URL,
or ``{"url": ...}``. A video is either one such reference to a video file (decoded with PyAV and
sampled uniformly) or a list of image references, one per frame.

Every decoded video carries metadata (frame count, fps, duration). Qwen3-VL-style processors need it:
without metadata they assume 24 fps and resample to 2 fps, which collapses a short clip to 2 frames.
"""

from __future__ import annotations

import base64
import binascii
import io
from dataclasses import dataclass, field
from typing import Any

import httpx
import numpy as np
from PIL import Image

from jevlike_infer.errors import RequestError


@dataclass(frozen=True)
class MediaConfig:
    max_bytes: int = 50 * 1024 * 1024
    fetch_timeout: float = 20.0
    allow_urls: bool = True
    video_frames: int = 16
    frame_list_fps: float = 2.0


@dataclass
class Video:
    """Sampled RGB frames ``(T, H, W, 3)`` plus the metadata of the sampled clip."""

    frames: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def load_bytes(ref: Any, config: MediaConfig) -> bytes:
    if isinstance(ref, dict):
        ref = ref.get("url") or ref.get("data")
    if not isinstance(ref, str) or not ref:
        raise RequestError("media item must be a URL, data URI, base64 string, or {url}")
    if ref.startswith(("http://", "https://")):
        if not config.allow_urls:
            raise RequestError("media URLs are disabled on this server; send data URIs or base64")
        try:
            with httpx.Client(timeout=config.fetch_timeout, follow_redirects=True) as client:
                response = client.get(ref)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RequestError(f"failed to fetch media {ref}: {exc}") from exc
        data = response.content
    else:
        payload = ref.split(",", 1)[1] if ref.startswith("data:") else ref
        try:
            data = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RequestError("media item is not valid base64") from exc
    if len(data) > config.max_bytes:
        raise RequestError(f"media item exceeds {config.max_bytes} bytes")
    return data


def decode_image(ref: Any, config: MediaConfig) -> Image.Image:
    data = load_bytes(ref, config)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise RequestError(f"cannot decode image: {exc}") from exc


def clip_metadata(frames: np.ndarray, fps: float) -> dict[str, Any]:
    count, height, width = frames.shape[:3]
    return {
        "total_num_frames": count,
        "fps": fps,
        "duration": count / fps,
        "frames_indices": list(range(count)),
        "width": width,
        "height": height,
    }


def decode_video(ref: Any, config: MediaConfig) -> Video:
    if isinstance(ref, list):
        return _video_from_frames(ref, config)
    return _video_from_file(load_bytes(ref, config), config)


def _video_from_frames(refs: list[Any], config: MediaConfig) -> Video:
    if not refs:
        raise RequestError("video frame list must not be empty")
    frames = [np.asarray(decode_image(ref, config)) for ref in refs]
    if len({frame.shape for frame in frames}) != 1:
        raise RequestError("all frames of a video must have the same size")
    stacked = np.stack(frames)
    return Video(stacked, clip_metadata(stacked, config.frame_list_fps))


def _video_from_file(data: bytes, config: MediaConfig) -> Video:
    import av

    try:
        with av.open(io.BytesIO(data)) as container:
            stream = container.streams.video[0]
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            rate = float(stream.average_rate) if stream.average_rate else None
    except Exception as exc:
        raise RequestError(f"cannot decode video: {exc}") from exc
    if not frames:
        raise RequestError("video has no frames")
    indices = np.linspace(0, len(frames) - 1, num=min(config.video_frames, len(frames))).round().astype(int)
    sampled = np.stack([frames[index] for index in indices])
    duration = duration or len(frames) / (rate or 24.0)
    return Video(sampled, clip_metadata(sampled, len(sampled) / duration))
