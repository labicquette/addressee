from typing import Any, Mapping
import torch
import torch.nn as nn
from torchmetrics.functional.classification import binary_f1_score, f1_score, multiclass_recall, multiclass_precision, multiclass_confusion_matrix
import lightning as pl
import pandas as pd 
import seaborn as sns

from addressee.utils.schedulers import TriStageLRScheduler, build_scheduler
from addressee.utils.modeling import ConvolutionSettings
from addressee.utils.config import namespace_to_dict
from addressee.data.dataloaders import binary_classes,ternary_classes
import math

from typing import Literal, Optional, Tuple, Iterable
from .utils import load_wav2vec2_model


run_id2model_id = {
    "bbh1padded0": "BabyHuBERT 1",
    "bbh2padded0":"BabyHuBERT 2",
    "hubertbasepadded0":"HuBERT-base",
    "hubertlargepadded0":"HuBERT-large",
}


class Wav2VecFinetune(pl.LightningModule):
    def __init__(
        self,
        config,
        train: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.label_encoder = binary_classes
        if train:
            self.wav2vec2 = load_wav2vec2_model(self.config.model_id, self.config.model_checkpoint)
            self.wav2vec2.train()

        self.padding_attention_mask = self.config.train.mask_padding_attention
        if "wavlm" in self.config.model_id:
            self.padding_attention_mask = False

        if self.config.model_id == "hubert_large":
            feature_size = 1024
        else:
            feature_size = 768

        if "xlsr" in self.config.model_id:
            self.xlsr = True
            feature_size = 1024
        else: 
            self.xlsr = False

        # NOTE - freeze CNN encoder
        for p in self.wav2vec2.feature_extractor.parameters():
            p.requires_grad = False

        # NOTE - freeze transformer encoder, opt.
        if self.config.freeze_encoder:
            for p in self.wav2vec2.encoder.parameters():
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

        self.automatic_optimization = False
        self.scaler = torch.amp.GradScaler("cuda")
        self.nan_loss_count = 0.0
        self.distributed = False
        self.clip_norm = 10.0
        self.max_penalty = 0.2
        self.save_hyperparameters(namespace_to_dict(self.config))

        self.preds = {"test":[],"heldout":[]}
        self.targets = {"test":[],"heldout":[]}
        self.context_size = int(self.config.context_size)

    def forward(self, x: torch.Tensor, lengths, mask):
        x = x.squeeze(1)
        lengths = None
        # xlsr needs an extra layer norm
        if self.xlsr:
            x = nn.functional.layer_norm(x, x.shape)
               
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
            if self.context_size == 0:
                x = x.mean(dim=1)
            else:
                mask_sum =  mask.sum(dim=1)
                x = (x * mask).sum(dim=1) / mask_sum


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

        autocast = True
        if self.xlsr:
            autocast=False
        
        with torch.amp.autocast("cuda", enabled=autocast):
                x, y_target, lengths, mask = batch
                y_preds = self.forward(x, lengths=lengths, mask=mask)

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
        x, y_target, lengths, mask = batch
        y_preds = self.forward(x, lengths=lengths, mask=mask)

        n_labels = len(self.label_encoder.keys())
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
        x, y_target, lengths, mask = batch
        y_preds = self.forward(x, lengths=lengths, mask=mask)

        n_labels = len(self.label_encoder.keys())
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
        x, y_target, lengths, mask = batch
        y_preds = self.forward(x, lengths=lengths, mask=mask)
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





