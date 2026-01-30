from .hubert.modeling_hubert import HubertFinetune
from .wav2vec.modeling_wav2vec import Wav2VecFinetune

Models = {
    "hubert": HubertFinetune,
    "wav2vec" : Wav2VecFinetune

}

Id_to_Model = {}

__all__ = [
    "HubertFinetune",
    "Wav2VecFinetune"
    "Models",
]
