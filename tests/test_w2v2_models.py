import torch

from addressee.models.wav2vec.utils import load_wav2vec2_model


def test_wav2vec2_base_bare():
    w2v2_model = load_wav2vec2_model("wav2vec2_base")

    x_t = torch.zeros(4, 64_000)
    y_t, _ = w2v2_model(x_t)
    assert y_t.shape == (4, 199, 768)


def test_wav2vec2_large_bare():
    """Load a bare model and forward tensor"""
    w2v2_model = load_wav2vec2_model("wav2vec2_large")

    x_t = torch.zeros(4, 64_000)
    y_t, _ = w2v2_model(x_t)
    assert y_t.shape == (4, 199, 1024)


def test_wav2vec2_xlsr53_bare():
    """Load a bare model and forward tensor"""
    w2v2_model = load_wav2vec2_model("wav2vec2_xlsr53")

    x_t = torch.zeros(4, 64_000)
    y_t, _ = w2v2_model(x_t)
    assert y_t.shape == (4, 199, 1024)


def test_wav2vec2_LL4300():
    w2v2_model = load_wav2vec2_model(
        "wav2vec2_ll4300", "models/wav2vec2_ll4300_torchaudio.pt"
    )

    x_t = torch.zeros(4, 64_000)
    y_t, _ = w2v2_model(x_t)
    assert y_t.shape == (4, 199, 768)
