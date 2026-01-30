from pathlib import Path

import torch
import torchaudio
from torch.nn import Module
from torchaudio.models import hubert_pretrain_base


def load_wav2vec(path: Path | str):
    path = Path(path)

    print("loading : ",path)
    model = hubert_pretrain_base(num_classes=500)
    if path.exists():
        print("loaded custom path", path)
        model = _load_state(model, path)

    else:
        if str(path) == "wavlm_base_plus":
            print("loading wavlm_base_plus")
            bundle = torchaudio.pipelines.WAVLM_BASE_PLUS

        if str(path) == "wavlm_base":
            print("loading wavlm_base")
            bundle = torchaudio.pipelines.WAVLM_BASE

        if str(path) == "wav2vec2_base":
            print("loading wav2vec2_base")
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
        
        if str(path) == "wav2vec2_xlsr":
            print("loading wav2vec2_xlsr")
            bundle = torchaudio.pipelines.WAV2VEC2_XLSR53

        wav2vec2 = bundle.get_model()
        wav2vec2.train()

    return wav2vec2


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


