from pathlib import Path

import torch
import torchaudio
from torch.nn import Module
from torchaudio.models import hubert_pretrain_base


def load_hubert(path: Path | str):
    path = Path(path)

    print("loading : ",path)
    model = hubert_pretrain_base(num_classes=500)
    if path.exists():
        print("loaded custom path", path)
        model = _load_state(model, path)

    else:
        if "base" in str(path):
            print("loading HuBERT-base")
            bundle = torchaudio.pipelines.HUBERT_BASE
            wav2vec2 = bundle.get_model()
            model.wav2vec2 = wav2vec2
        if "large" in str(path):
            print("loading HuBERT-large")
            bundle = torchaudio.pipelines.HUBERT_LARGE
            wav2vec2 = bundle.get_model()
            model.wav2vec2 = wav2vec2.model
    model.wav2vec2.train()
    return model.wav2vec2


def _load_state(model: Module, checkpoint_path: Path, device="cpu") -> Module:
    """Load weights from HuBERTPretrainModel checkpoint into hubert_pretrain_base model.
    Args:
        model (Module): The hubert_pretrain_base model.
        checkpoint_path (Path): The model checkpoint.
        device (torch.device, optional): The device of the model. (Default: ``torch.device("cpu")``)

    Returns:
        (Module): The pretrained model.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {
        k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return model


