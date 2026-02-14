from pathlib import Path
from typing import Literal, Any

import torch
import torchaudio
from torchaudio.models.wav2vec2 import wav2vec2_base


# https://docs.pytorch.org/audio/stable/generated/torchaudio.models.Wav2Vec2Model.html#torchaudio.models.Wav2Vec2Model
W2V2_MODELS = [
    "wav2vec2_base",
    "wav2vec2_large",
    "wav2vec2_xlsr53",
    "wav2vec2_ll4300",
    "wavlm"
]

def load_wav2vec2_model(
    model_id: Literal[
        "wav2vec2_base", "wav2vec2_large", "wav2vec2_xlsr53", "wav2vec2_ll4300"
    ],
    model_checkpoint: Path | None = None,
) -> Any:
    """Load a pre-trained wav2vec 2.0 model"""
    match model_id:
        case "wav2vec2_base":
            return torchaudio.pipelines.WAV2VEC2_BASE.get_model()
        case "wav2vec2_large":
            return torchaudio.pipelines.WAV2VEC2_LARGE.get_model()
        case "wav2vec2_xlsr53":
            return torchaudio.pipelines.WAV2VEC2_XLSR53.get_model().model
        case "wavlm":
            return torchaudio.pipelines.WAVLM_BASE_PLUS.get_model()
        case "wav2vec2_ll4300":
            assert model_checkpoint is not None
            model = wav2vec2_base()
            model.load_state_dict(torch.load(model_checkpoint, map_location="cpu"))
            return model
        case _:
            raise ValueError(
                f"The `model_id` value is invalid, please select one of: {W2V2_MODELS}."
            )
