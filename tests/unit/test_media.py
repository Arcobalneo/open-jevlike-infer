import base64
import io

import av
import numpy as np
import pytest
from PIL import Image

from jevlike_infer.errors import RequestError
from jevlike_infer.media import MediaConfig, decode_image, decode_video

CONFIG = MediaConfig()


def png(color="red", size=(32, 32), mode="RGB") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def data_uri(raw: bytes, mime="image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


def mp4(frame_count=24, fps=24) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 32, "yuv420p"
        for i in range(frame_count):
            frame = np.full((32, 64, 3), i * 10 % 255, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


@pytest.mark.parametrize(
    "ref",
    [data_uri(png()), base64.b64encode(png()).decode(), {"url": data_uri(png())}, {"data": data_uri(png())}],
)
def test_image_reference_forms(ref):
    image = decode_image(ref, CONFIG)
    assert image.mode == "RGB" and image.size == (32, 32)


def test_rgba_is_converted():
    assert decode_image(data_uri(png((255, 0, 0, 128), mode="RGBA")), CONFIG).mode == "RGB"


@pytest.mark.parametrize(
    ("ref", "message"),
    [
        ("!!!not base64!!!", "base64"),
        (base64.b64encode(b"hello").decode(), "cannot decode image"),
        (123, "media item"),
        ({"other": 1}, "media item"),
    ],
)
def test_image_errors(ref, message):
    with pytest.raises(RequestError, match=message):
        decode_image(ref, CONFIG)


def test_size_limit():
    with pytest.raises(RequestError, match="exceeds"):
        decode_image(data_uri(png()), MediaConfig(max_bytes=10))


def test_urls_can_be_disabled():
    with pytest.raises(RequestError, match="disabled"):
        decode_image("https://example.com/a.png", MediaConfig(allow_urls=False))


def test_frame_list_keeps_every_frame_and_has_metadata():
    video = decode_video([data_uri(png("blue", (40, 20)))] * 6, CONFIG)
    assert video.frames.shape == (6, 20, 40, 3)
    meta = video.metadata
    assert meta["total_num_frames"] == 6 and meta["fps"] == 2.0 and meta["duration"] == 3.0
    assert meta["frames_indices"] == list(range(6)) and (meta["width"], meta["height"]) == (40, 20)


def test_frame_list_errors():
    with pytest.raises(RequestError, match="empty"):
        decode_video([], CONFIG)
    with pytest.raises(RequestError, match="same size"):
        decode_video([data_uri(png(size=(8, 8))), data_uri(png(size=(4, 4)))], CONFIG)


def test_video_file_is_sampled_with_real_fps():
    video = decode_video(data_uri(mp4(frame_count=48, fps=24), "video/mp4"), MediaConfig(video_frames=16))
    assert video.frames.shape[0] == 16
    assert video.metadata["duration"] == pytest.approx(2.0, rel=0.1)
    assert video.metadata["fps"] == pytest.approx(8.0, rel=0.1)  # 16 sampled frames over ~2 s


def test_short_video_keeps_all_frames():
    assert decode_video(data_uri(mp4(frame_count=5), "video/mp4"), CONFIG).frames.shape[0] == 5


def test_bad_video():
    with pytest.raises(RequestError, match="cannot decode video"):
        decode_video(base64.b64encode(b"not a video").decode(), CONFIG)
