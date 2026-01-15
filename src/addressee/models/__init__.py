from .hubert.modeling_hubert import HubertFinetune

Models = {
    "hubert": HubertFinetune
}

Id_to_Model = {}

__all__ = [
    "HubertFinetune"
    "Models",
]
