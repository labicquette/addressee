import torch
from typing import Literal


def get_parameter_table(model: torch.nn.Module):
    m_length = max(len(name) for name, _ in model.named_parameters()) + 2
    total_params = 0
    total_trainable_params = 0

    print(f"{'Name':<{m_length}} | {'Params':<10} | {'Trainable params':<10}")
    print("-" * (m_length + 32))
    for name, p in model.named_parameters():
        np = p.numel()
        ntp = p.numel() if p.requires_grad else 0

        total_params += np
        total_trainable_params += ntp

        print(f"{name:<{m_length}} | {np:<10} | {ntp:<10}")

    # TODO - add percent trainable
    print(total_params, total_trainable_params)


def get_metric(metric: str) -> tuple[Literal["min", "max"], str]:
    match metric:
        case "loss":
            return "min", "val/loss"
        case "f1_score":
            return "max", "val/f1_score"
        case "auroc":
            return "max", "val/auroc"
        case _:
            raise ValueError(
                f"metric '{metric}' is not supported, please use 'loss', 'auroc' or 'f1_score'."
            )
