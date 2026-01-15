import torch
from torch.optim.optimizer import Optimizer

import math

import torch
from torch import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, LinearLR, LRScheduler, SequentialLR


class TriStageLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear learning rate scheduler with warmup, hold, and decay."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        hold_updates: int,
        decay_updates: int,
        init_lr_scale: float = 0.01,
        final_lr_scale: float = 0.05
    ):
        self.warmup_updates = warmup_updates
        self.hold_updates = hold_updates
        self.decay_updates = decay_updates
        self.init_lr_scale = init_lr_scale
        self.final_lr_scale = final_lr_scale

        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                base_lr * (self.init_lr_scale + self._step_count / self.warmup_updates * (1 - self.init_lr_scale))
                for base_lr in self.base_lrs
            ]
        elif self.warmup_updates < self._step_count <= (self.warmup_updates + self.hold_updates):
            return list(self.base_lrs)
        elif self._step_count <= (self.warmup_updates + self.hold_updates + self.decay_updates):
            return [
                base_lr
                * math.exp(
                    math.log(self.final_lr_scale)
                    * (self._step_count - self.warmup_updates - self.hold_updates)
                    / self.decay_updates
                )
                for base_lr in self.base_lrs
            ]
        else:
            return [base_lr * self.final_lr_scale for base_lr in self.base_lrs]




def build_scheduler(opt: Optimizer, cfg) -> LRScheduler:
    init_lr_scale = cfg.init_lr_scale if cfg.warmup_steps > 0 else 1
    decay: LRScheduler
    if cfg.scheduler == "tristage":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.hold_steps)
        decay = LambdaLR(opt, lambda step: math.exp(math.log(cfg.final_lr_scale) * step / cfg.decay_steps))
        return SequentialLR(opt, [warmup, hold, decay], [cfg.warmup_steps, cfg.hold_steps + cfg.warmup_steps])
    if cfg.scheduler == "cosine":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        decay = CosineAnnealingLR(opt, cfg.max_steps - cfg.warmup_steps, cfg.final_lr_scale * cfg.lr)
        return SequentialLR(opt, [warmup, decay], [cfg.warmup_steps])
    if cfg.scheduler == "rsqrt":
        warmup = LinearLR(
            opt,
            start_factor=init_lr_scale,
            end_factor=1 / math.sqrt(1 + cfg.rsqrt_shift / cfg.rsqrt_timescale),
            total_iters=cfg.warmup_steps,
        )
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.hold_steps)
        decay = LambdaLR(opt, lambda step: 1 / math.sqrt(1 + (step + cfg.rsqrt_shift) / cfg.rsqrt_timescale))
        return SequentialLR(opt, [warmup, hold, decay], [cfg.warmup_steps, cfg.hold_steps + cfg.warmup_steps])
    if cfg.scheduler == "constant":
        warmup = LinearLR(opt, start_factor=init_lr_scale, total_iters=cfg.warmup_steps)
        hold = LinearLR(opt, start_factor=1.0, total_iters=cfg.max_steps)
        return SequentialLR(opt, [warmup, hold], [cfg.warmup_steps])
    raise ValueError(f"Unknown scheduler: {cfg.scheduler}")