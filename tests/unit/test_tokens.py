from jevlike_infer.models.clef_flash.tokens import VisionTokens, collapse_media_placeholders

T = VisionTokens(image_pad=1, video_pad=2, vision_start=3, vision_end=4)
TS = 50  # stands for timestamp text tokens such as "<0.2 seconds>"


def test_text_only_is_unchanged():
    assert collapse_media_placeholders([10, 11, 12], T) == [10, 11, 12]


def test_image_keeps_one_pad():
    assert collapse_media_placeholders([10, 3, 1, 1, 1, 1, 4, 11], T) == [10, 3, 1, 4, 11]


def test_two_images():
    ids = [3, 1, 1, 4, 3, 1, 1, 1, 4, 9]
    assert collapse_media_placeholders(ids, T) == [3, 1, 4, 3, 1, 4, 9]


def test_qwen3_vl_video_with_timestamps_collapses_to_one_placeholder():
    # <vs> (<t> <vs> pad*3 <ve>) x2 <ve>
    ids = [10, 3, TS, 51, 3, 2, 2, 2, 4, TS, 52, 3, 2, 2, 2, 4, 4, 11]
    assert collapse_media_placeholders(ids, T) == [10, 3, 2, 4, 11]


def test_video_without_timestamps():
    assert collapse_media_placeholders([3, 2, 2, 2, 4, 9], T) == [3, 2, 4, 9]


def test_image_then_video():
    ids = [3, 1, 1, 4, 3, TS, 3, 2, 2, 4, 4, 9]
    assert collapse_media_placeholders(ids, T) == [3, 1, 4, 3, 2, 4, 9]


def test_pad_only_collapse_is_not_enough_for_video():
    """The naive approach (dedupe repeated pads) leaves timestamps and nested markers behind."""
    ids = [3, TS, 3, 2, 2, 4, TS, 3, 2, 2, 4, 4]
    naive = [t for i, t in enumerate(ids) if not (t in (1, 2) and i and ids[i - 1] == t)]
    assert naive != [3, 2, 4]
    assert collapse_media_placeholders(ids, T) == [3, 2, 4]
