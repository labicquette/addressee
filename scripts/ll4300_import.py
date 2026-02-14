# /// script
# requires-python = "==3.10"
# dependencies = [
#     "fairseq>=0.12.2",
#     "torch>=2.6.0",
# ]
# ///
from pathlib import Path

import torch


def fairseq_to_torchaudio(
    model_path: Path, output: str = "models/wav2vec2_ll4300_torchaudio.pt"
):
    import fairseq
    from torchaudio.models.wav2vec2.utils import import_fairseq_model

    assert model_path.exists()
    original, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    torchaudio_model = import_fairseq_model(original[0])

    # Save clean torchaudio-only checkpoint
    torch.save(torchaudio_model.state_dict(), output)
    print("fairseq w2v2 model successfully converted to torchaudio")


def test_forward(torchaudio_model_path: Path):
    from torchaudio.models.wav2vec2 import wav2vec2_base

    model = wav2vec2_base()
    model.load_state_dict(torch.load(torchaudio_model_path, map_location="cpu"))

    out, _ = model.extract_features(torch.ones(8, 64_000))[0]
    assert len(out) == 12 and out[0].shape == (8, 199, 768)


if __name__ == "__main__":
    fairseq_w2v2 = Path("models/checkpoint_best.pt")
    torchaudio_w2v2 = Path("models/wav2vec2_ll4300_torchaudio.pt")

    fairseq_to_torchaudio(fairseq_w2v2, torchaudio_w2v2)
    test_forward(torchaudio_w2v2)
