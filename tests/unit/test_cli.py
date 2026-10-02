import pytest

from jevlike_infer.cli import build_parser, main, settings_from_args


def test_models_command(capsys):
    main(["models"])
    assert "clef-flash\tbackends: vllm, hf" in capsys.readouterr().out


def test_defaults_pick_first_backend():
    settings = settings_from_args(build_parser().parse_args(["serve", "--model-path", "/m"]))
    assert settings.backend == "vllm" and settings.model_name == "clef-flash" and settings.warmup


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("JEVLIKE_MODEL_PATH", "/weights")
    monkeypatch.setenv("JEVLIKE_PORT", "9000")
    monkeypatch.setenv("JEVLIKE_NO_MEDIA_URLS", "true")
    settings = settings_from_args(build_parser().parse_args(["serve"]))
    assert settings.model_path == "/weights" and settings.port == 9000 and not settings.media.allow_urls


def test_unknown_backend():
    with pytest.raises(SystemExit):
        settings_from_args(build_parser().parse_args(["serve", "--model-path", "/m", "--backend", "tensorrt"]))
