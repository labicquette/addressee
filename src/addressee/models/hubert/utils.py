from pathlib import Path

import torch
import torchaudio
from torch.nn import Module
from torchaudio.models import hubert_pretrain_base


def load_hubert(module):
    path = Path(module.config.model_id)

    if path.exists():
        state_dict = torch.load(path, map_location="cpu")
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
        }
        # Load directly into the HubertFinetune module (restores wav2vec2 + classifier)
        module.load_state_dict(state_dict, strict=False)
        module.wav2vec2 = _init_backbone(module.config.model_id)
        # Now reload so wav2vec2 weights are also restored
        module.load_state_dict(state_dict, strict=False)
    else:
        module.wav2vec2 = _init_backbone(module.config.model_id)



def _init_backbone(model_id: str):
    """Load a pretrained HuBERT backbone from torchaudio."""
    model = hubert_pretrain_base(num_classes=500)
    if "base" in model_id:
        print("loading HuBERT-base")
        bundle = torchaudio.pipelines.HUBERT_BASE
        model.wav2vec2 = bundle.get_model()
    elif "large" in model_id:
        print("loading HuBERT-large")
        bundle = torchaudio.pipelines.HUBERT_LARGE
        model.wav2vec2 = bundle.get_model()
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


