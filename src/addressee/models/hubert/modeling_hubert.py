from typing import Any, Mapping
import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_f1_score, f1_score, multiclass_recall
import lightning as pl

from addressee.utils.schedulers import TriStageLRScheduler, build_scheduler
from addressee.utils.modeling import ConvolutionSettings
from addressee.utils.config import namespace_to_dict
from addressee.data.dataloaders import binary_classes,ternary_classes
import math

from typing import Literal, Optional, Tuple, Iterable
from .utils import load_hubert


class HubertFinetune(pl.LightningModule):
    def __init__(
        self,
        config,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_encoder = binary_classes
        if train:
            self.wav2vec2 = load_hubert(self.config.model_id)
        # else:
        #     from torchaudio.models import hubert_pretrain_base

        #     model = hubert_pretrain_base(num_classes=500)
        #     self.wav2vec2 = model.wav2vec2

        if self.config.model_id == "hubert_large" :
            feature_size = 1024
        else:
            feature_size = 768

        # NOTE - freeze CNN encoder
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

        # NOTE - freeze transformer encoder, opt.
        if self.config.freeze_encoder:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False
            
        # reduction mechanism - learnable or non-learnable weights
        if self.config.reduction == "weighted":

            self.enc_layers_to_use = list(
                range(len(self.wav2vec2.encoder.transformer.layers))
            )

            self.layer_weights = nn.Parameter(
                torch.ones(len(self.enc_layers_to_use)) / len(self.enc_layers_to_use)
            )


        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(in_features=feature_size, out_features=3)

        self.conv_settings = ConvolutionSettings(
            kernels=(10, 3, 3, 3, 3, 2, 2),
            strides=(5, 2, 2, 2, 2, 2, 2),
            paddings=(0, 0, 0, 0, 0, 0, 0),
        )

        self.automatic_optimization = False
        self.scaler = torch.amp.GradScaler("cuda")
        self.nan_loss_count = 0.0
        self.distributed = False
        self.clip_norm = 10.0
        self.max_penalty = 0.2
        self.save_hyperparameters(namespace_to_dict(self.config))


    def forward(self, x: torch.Tensor, mask):
        x = x.squeeze(1)
        with torch.no_grad():
            x, lengths = self.wav2vec2.feature_extractor(x, None)
        if self.config.freeze_encoder:
            with torch.no_grad():
                hidden_states = self.wav2vec2.encoder.extract_features(
                    x, lengths, num_layers=None
                )
        else:
            hidden_states = self.wav2vec2.encoder.extract_features(
                x, lengths, num_layers=None
            )

        if self.config.reduction:
            # hidden_states = torch.stack(hidden_states, dim=0)
            # weights = self.layer_weights.view(-1, 1, 1, 1)
            # x = (weights * hidden_states).sum(dim=0)

            hidden_states = torch.stack(hidden_states, dim=0)
            #print(hidden_states.shape)
            layer_weights = self.layer_weights[:, None, None, None]
            x = torch.sum(layer_weights * hidden_states, dim=0)
        else:
            x = hidden_states[-1]

        if hasattr(self.config, "pool"):
            #print(x.shape)
            #x = x[mask[0]:mask[1]].mean(dim=1)  # (B, H)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        # careful if not pooled, whole sequence of frames
        else:
            raise NotImplementedError(
                f"Transformer layer for dynamic frame sequence is not implemented"
            )
            #here transformers logic

        # here x should be a single 768 representation
        x = self.dropout(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):

        opt = self.optimizers()
        opt.zero_grad()

        with torch.amp.autocast("cuda", enabled=True):
            x, y_target, mask = batch

            if self.config.padding: 

            # x = batch["x"]
            # y_target = batch["y"]
            y_preds = self.forward(x, mask=mask)
            # reduce first 2 dimensions (batch and windows can be merged)
            #n_labels = len(self.label_encoder.keys())
            #labels = self.label_encoder.keys()
            #y_target = y_target.view(-1, n_labels)
            # (batch * n_windows) - flattened, usefull when slicing target vector at the end
            #y_preds = y_pred_heads.view(-1, n_labels)

            loss = torch.nn.functional.cross_entropy(
                    input=y_preds,
                    target=y_target
                )
            self.log(
                "train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, logger=True
            )
        self.scaler.scale(loss)

        self.manual_backward(loss)
        self.scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.wav2vec2.parameters(), self.clip_norm)
        self.log("Grad_norm", grad_norm, on_step=True, on_epoch=True)
        # optimization
        self.scaler.step(opt)
        sch = self.lr_schedulers()
        sch.step()
        self.scaler.update()

    def validation_step(self, batch, batch_idx):
        x, y_target, mask = batch
        # x = batch["x"]
        # y_target = batch["y"]
        y_preds = self.forward(x, mask=mask)

        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.label_encoder.keys())
        labels = self.label_encoder.keys()
        #y_target = y_target.view(-1, n_labels)
        # (batch * n_windows) - flattened, usefull when slicing target vector at the end
        #y_preds = {k: y_pred.view(-1) for k, y_pred in y_pred_heads.items()}
        #y_preds = y_pred_heads.view(-1, n_labels)

        # NOTE - loss computation
        if (
            self.config.train.validation_metric == "loss"
            or "loss" in self.config.train.extra_val_metrics
        ):
            # head_losses = {
            #     k: torch.nn.functional.binary_cross_entropy_with_logits(
            #         input=y_pred, target=y_target[..., i], weight=self.weights
            #     )
            #     for i, (k, y_pred) in enumerate(y_preds.items())
            # }
            loss = torch.nn.functional.cross_entropy(
                    input=y_preds,
                    target=y_target
                )
            self.log(
                "val/loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # NOTE - f1 score
        if (
            self.config.train.validation_metric == "f1_score"
            or "f1_score" in self.config.train.extra_val_metrics
        ):
            # for i,label in enumerate(labels):
            #     head_f1_i = binary_f1_score(
            #         preds=y_preds[:, i],
            #         target=y_target[:, i],
            #         threshold=0.5,
            #     )
            #     self.log(
            #         f"val/f1_{label}",
            #         head_f1_i,
            #         on_step=True,
            #         on_epoch=True,
            #         prog_bar=True,
            #         logger=True,
            #     )

            whole_f1 = f1_score(
                    preds=y_preds,
                    target=y_target,
                    task="multiclass",
                    num_classes=len(labels)
                )
            self.log(
                    f"val/f1_score",
                    whole_f1,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            

            uar_recall = multiclass_recall(
                preds=y_preds,
                target=y_target,
                num_classes=len(labels),
                average="macro"
                )
            self.log(
                    f"val/uar_recall",
                    uar_recall,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

            classes_f1 = f1_score(
                    preds=y_preds,
                    target=y_target,
                    task="multiclass",
                    num_classes=len(labels),
                    average=None
                )
            classes_f1 = dict(zip(labels, classes_f1.tolist()))
            for head_name, head_loss in classes_f1.items():
                self.log(
                    f"val/F1_{head_name}",
                    head_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                )

    def test_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y_target, mask = batch
        # x = batch["x"]
        # y_target = batch["y"]
        y_preds = self.forward(x, mask=mask)

        # reduce first 2 dimensions (batch and windows can be merged)
        n_labels = len(self.label_encoder.keys())
        labels = self.label_encoder.keys()
        #y_target = y_target.view(-1, n_labels)
        # (batch * n_windows) - flattened, usefull when slicing target vector at the end
        #y_preds = {k: y_pred.view(-1) for k, y_pred in y_pred_heads.items()}
        #y_preds = y_pred_heads.view(-1, n_labels)
        
        dataloader_names = {0: "test", 1: "heldout"}
        dataset_name = dataloader_names.get(dataloader_idx, f"dataset_{dataloader_idx}")
        
        # NOTE - f1 score
        if (
            self.config.train.validation_metric == "f1_score"
            or "f1_score" in self.config.train.extra_val_metrics
        ):

            whole_f1 = f1_score(
                    preds=y_preds,
                    target=y_target,
                    task="multiclass",
                    num_classes=len(labels)
                )
            self.log(
                    f"{dataset_name}/f1_score",
                    whole_f1,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    add_dataloader_idx=False
                )
            

            uar_recall = multiclass_recall(
                preds=y_preds,
                target=y_target,
                num_classes=len(labels),
                average="macro"
                )
            self.log(
                    f"{dataset_name}/uar_recall",
                    uar_recall,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    add_dataloader_idx=False
                )

            classes_f1 = f1_score(
                    preds=y_preds,
                    target=y_target,
                    task="multiclass",
                    num_classes=len(labels),
                    average=None
                )
            classes_f1 = dict(zip(labels, classes_f1.tolist()))
            for head_name, head_loss in classes_f1.items():
                self.log(
                    f"{dataset_name}/F1_{head_name}",
                    head_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    add_dataloader_idx=False
                )

    def configure_optimizers(self):
        self.optimizer_finetune = torch.optim.AdamW(
            list(self.wav2vec2.parameters()) + list(self.classifier.parameters()),
            lr=self.config.train.lr,
            fused=True
        )
        mode, monitor = get_metric(self.config.train.validation_metric)
        #self.lr_scheduler_finetune = TriStageLRScheduler(self.optimizer_finetune, warmup_updates=4000,hold_updates=40000, decay_updates=20000)
        self.lr_scheduler_finetune = build_scheduler(self.optimizer_finetune, self.config.train.optim)
        #self.lr_scheduler_finetune = ReduceLROnPlateau(
        #        self.optimizer_finetune, mode=mode, patience=self.config.train.scheduler.patience
        #     )
        return (
            {"optimizer": self.optimizer_finetune,
             "lr_scheduler": {
                 "scheduler": self.lr_scheduler_finetune
                },
             "monitor": monitor,
             },
        )
    

    def state_dict(self, *args, **kwargs):
        """Custom state_dict that excludes the whisper encoder."""
        state_dict = super().state_dict(*args, **kwargs)
        # Remove all entries starting with 'w_encoder.'
        keys_to_remove = [k for k in state_dict.keys() if k.startswith("encoder.")]
        for k in keys_to_remove:
            del state_dict[k]
        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Custom load_state_dict that doesn't require whisper encoder weights."""
        return super().load_state_dict(state_dict, strict=False, assign=assign)
    



def get_metric(metric: str) -> tuple[Literal["min", "max"], str]:
        match metric:
            case "loss":
                return "min", "val/loss"#/dataloader_idx_0"
            case "f1_score":
                return "max", "val/f1_score"
            case "auroc":
                return "max", "val/auroc"
            case _:
                raise ValueError(
                    f"metric '{metric}' is not supported, please use 'loss', 'auroc' or 'f1_score'."
                )





