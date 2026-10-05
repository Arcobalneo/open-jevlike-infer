from jevlike_infer.models.clef_flash.vllm_backend import engine_max_model_len


def test_engine_window_has_one_token_of_headroom():
    # vLLM reserves one sampled-token slot per scheduling step, also for pooling requests, so a prompt
    # that fills max_model_len exactly is never fully scheduled and the engine spins forever.
    assert engine_max_model_len(16384) == 16385


def test_engine_window_scales_with_max_length():
    assert engine_max_model_len(4096) - 4096 == engine_max_model_len(16384) - 16384 == 1
