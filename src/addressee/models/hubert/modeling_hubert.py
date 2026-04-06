from typing import Any, Mapping
import torch
import torch.nn as nn
from torchmetrics.functional.classification import f1_score, multiclass_recall, multiclass_precision, multiclass_confusion_matrix
import lightning as pl
import pandas as pd 
import seaborn as sns
from addressee.utils.schedulers import build_scheduler
from addressee.utils.config import namespace_to_dict
from addressee.data.dataloaders import binary_classes
from typing import Literal
from .utils import load_hubert


run_id2model_id = {
    "bbh1padded0": "BabyHuBERT 1",
    "bbh2padded0":"BabyHuBERT 2",
    "hubertbasepadded0":"HuBERT-base",
    "hubertlargepadded0":"HuBERT-large",
}


class HubertFinetune(pl.LightningModule):
    def __init__(
        self,
        config,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_encoder = binary_classes

        if self.config.model_id == "hubert_large" :
            feature_size = 1024
        else:
            feature_size = 768


        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(in_features=feature_size, out_features=3)
        from torchaudio.models import hubert_pretrain_base

        model = hubert_pretrain_base(num_classes=500)
        self.wav2vec2 = model.wav2vec2

        if train:
            load_hubert(self)
            

        # NOTE - freeze CNN encoder
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

        # NOTE - freeze transformer encoder, opt.
        if self.config.freeze_encoder:
            for p in self.wav2vec2.parameters():
                p.requires_grad = False

        self.automatic_optimization = False
        self.scaler = torch.amp.GradScaler("cuda")
        self.nan_loss_count = 0.0
        self.distributed = False
        self.clip_norm = 10.0
        self.max_penalty = 0.2
        self.save_hyperparameters(namespace_to_dict(self.config))

        self.preds = {"test":[],"heldout":[]}
        self.targets = {"test":[],"heldout":[]}
        self.context_size = float(self.config.context_size)

    def forward(self, x: torch.Tensor, mask):
        x = x.squeeze(1)
        lengths = None
        with torch.no_grad():
            x, lengths = self.wav2vec2.feature_extractor(x, lengths)
        if self.config.freeze_encoder:
            with torch.no_grad():
                hidden_states = self.wav2vec2.encoder.extract_features(
                    x, lengths, num_layers=None
                )
        else:
            hidden_states = self.wav2vec2.encoder.extract_features(
                x, lengths, num_layers=None
            )

        x = hidden_states[-1]

        # x : (B,768)
        if self.context_size == 0:
            x = x.mean(dim=1)
        else:
            mask_sum =  mask.sum(dim=1)
            x = (x * mask).sum(dim=1) / mask_sum
                
        # here x should be a single 768 representation
        x = self.dropout(x)
        return self.classifier(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=True):
            x, y_target, mask = batch
            y_preds = self.forward(x, mask=mask)
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
        y_preds = self.forward(x, mask=mask)
        labels = self.label_encoder.keys()

        # NOTE - loss computation
        if (
            self.config.train.validation_metric == "loss"
            or "loss" in self.config.train.extra_val_metrics
        ):
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

            whole_f1 = f1_score(
                    preds=y_preds,
                    target=y_target,
                    task="multiclass",
                    num_classes=len(labels),
                    average="macro"
                )
            self.log(
                    f"val/f1_score",
                    whole_f1,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True
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
            
            precision = multiclass_precision(
                preds=y_preds,
                target=y_target,
                num_classes=len(labels),
                average="macro"
                )
            self.log(
                    f"val/precision",
                    precision,
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
        y_preds = self.forward(x, mask=mask)
        labels = self.label_encoder.keys()
        
        dataloader_names = {0: "test", 1: "heldout"}
        dataset_name = dataloader_names.get(dataloader_idx, f"dataset_{dataloader_idx}")
        
        self.preds[dataloader_names[dataloader_idx]] += [y_preds]
        self.targets[dataloader_names[dataloader_idx]] += [y_target]

        # NOTE - f1 score
        if (
            self.config.train.validation_metric == "f1_score"
            or "f1_score" in self.config.train.extra_val_metrics
        ):
            whole_f1 = f1_score(
                    preds=y_preds,
                    target=y_target,
                    task="multiclass",
                    num_classes=len(labels),
                    average="macro"
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
            precision = multiclass_precision(
                preds=y_preds,
                target=y_target,
                num_classes=len(labels),
                average="macro"
                )
            self.log(
                    f"{dataset_name}/precision",
                    precision,
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

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0):
        x, y_target, mask = batch
        y_preds = self.forward(x, mask=mask)
        return y_preds

    def on_test_epoch_end(self):
        if self.config.plots:
            for split in ["test", "heldout"]:
                confusion = multiclass_confusion_matrix(
                        preds=torch.cat(self.preds[split]),
                        target=torch.cat(self.targets[split]),
                        num_classes=len(self.label_encoder.keys()),
                        normalize="true"
                        )
                data = pd.DataFrame(confusion.cpu().numpy())
                plot = sns.heatmap(data, annot=True, xticklabels=self.label_encoder.keys(), yticklabels=self.label_encoder.keys())
                plot.set_title(split + " / " + run_id2model_id[self.config.run_id])
                plot.set(xlabel="pred", ylabel="target")
                fig = plot.get_figure()
                fig.savefig(f"/home/tcharlot/coml/addressee/confusion_matrix_{self.config.run_id}_{split}.png")
                fig.clf()
        return super().on_test_epoch_end()
    
    def configure_optimizers(self):
        self.optimizer_finetune = torch.optim.AdamW(
            list(self.wav2vec2.parameters()) + list(self.classifier.parameters()),
            lr=self.config.train.lr,
            fused=True
        )
        mode, monitor = get_metric(self.config.train.validation_metric)
        self.lr_scheduler_finetune = build_scheduler(self.optimizer_finetune, self.config.train.optim)
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





