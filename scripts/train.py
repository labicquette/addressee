import argparse
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Literal

import lightning as pl
import torch
import torch._dynamo.config
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.callbacks.progress.tqdm_progress import TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import pandas as pd 

from addressee.utils.logging import get_parameter_table, get_metric
from addressee.utils.config import load_config, save_config
from addressee.models import Models
from addressee.data.dataloaders import AddresseeDataloader

import os
import numpy as np
import random



id2label = {0: "ADS",
                  1: "KCDS",
                  2: "Other"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Config file to be loaded and used for the training.",
    )
    parser.add_argument(
        "--tags",
        nargs="*",
        default=[],
        help="Tags to be added to the wandb logging instance.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume training, pass in checkpoint path.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments",
        help="Output path of the model artifacts.",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default=None,
        help="Config file per experiment.",
    )
    parser.add_argument(
        "--run-id",
        "--id",
        type=str,
        help="ID of the run"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="hubert",
        help="Model Architecture.",
    )

    args, extra_args = parser.parse_known_args()


    if "SLURM_ARRAY_TASK_ID" in os.environ:
        rank = int(os.environ["SLURM_ARRAY_TASK_ID"])
    else :
        rank = 0

    if "SLURM_CPUS_PER_TASK" in os.environ:
        print("NUMBER OF CPUS PER TASK IS : ", int(os.environ["SLURM_CPUS_PER_TASK"]))
    else:
        print("DID NOT FIND SLURM CPUS COUNT")

    torch.set_float32_matmul_precision('highest')
    print("rank : ", rank)
    np.random.seed(rank) # if you're using numpy
    torch.manual_seed(rank)
    torch.cuda.manual_seed_all(rank)
    #torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    random.seed(rank)    

    config = load_config(args, extra_args)

    experiment_path = Path(args.output)
    if not experiment_path.exists():
        experiment_path.mkdir()

    save_path = experiment_path / args.run_id
    save_path.mkdir(parents=True, exist_ok=True)
    #config.save(save_path / "config.yml")

    save_config(config, save_path)

    chkp_path = save_path / "checkpoints"
    chkp_path.mkdir(parents=True, exist_ok=True)
    last_ckpt = chkp_path / "last.ckpt"



    mode, monitor = get_metric(config.train.validation_metric)

    print(config)
    model = Models[config.model_type](config)

    get_parameter_table(model)

    dm = AddresseeDataloader(dataset = "addressee",
                             dataset_path= config.data.dataset_path,
                             config= config)

    print("[log] - use WandbLogger")
    logger = WandbLogger(
        project=config.wandb.project,
        name=args.run_id,
        id=args.run_id.split("-")[-1],
        log_model=False,
        tags=args.tags,
        offline=config.wandb.offline,
        resume="must" if args.auto_resume and last_ckpt.exists() else None,  # "never",
    )
    # Allow val_change maybe not best idea but needed it for some reason
    # TODO
    logger.experiment.config.update(config, allow_val_change=True)
    #save_path = save_path.with_stem(save_path.stem + f"-{logger.experiment.id}")

    model_checkpoint = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        save_top_k=5,
        # every_n_epochs=1,
        save_last=True,
        dirpath=chkp_path,
        filename="epoch={epoch:02d}-val_loss={val/loss:.3f}",
        auto_insert_metric_name=False,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=mode,
        min_delta=0.0,
        patience=10,
        strict=True,
        verbose=False,
    )

    # NOTE - from scratch training and resume
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=config.train.max_epochs,
        logger=logger,
        callbacks=[
            model_checkpoint,
            early_stopping,
            LearningRateMonitor(),
            TQDMProgressBar(500 if "debug" not in config.data.dataset_path else 1),
        ],
        # profiler="advanced"
        profiler=config.train.profiler,
    )
    #model = torch.compile(model)

    
    trainer.validate(model=model, datamodule=dm)

    print(f"[log @ {datetime.now().strftime('%Y%m%d_%H%M')}] - Started training")
    if args.auto_resume and last_ckpt.exists():
        print("[log] - fit with resuming")
        trainer.fit(model, datamodule=dm, ckpt_path=last_ckpt)
    else:
        trainer.fit(model, datamodule=dm)


    # # NOTE - symlink to best model and to static best model (models/last/best.ckpt)
    best_link = chkp_path  / "best.ckpt"

    # if infer only heldout and test add comment until test
    if best_link.exists() or best_link.is_symlink():
        best_link.unlink()

    best_link.symlink_to(
        Path(model_checkpoint.best_model_path).absolute()
    )

    static_p = experiment_path / "last"
    static_p.mkdir(parents=True, exist_ok=True)
    bm_static_p = static_p / "best.ckpt"
    bm_static_p.unlink(missing_ok=True)
    bm_static_p.symlink_to(Path(model_checkpoint.best_model_path).absolute())

    print(f"[log] - best model score: {model_checkpoint.best_model_score}")
    print(f"[log] - best model path: {model_checkpoint.best_model_path}")


    print("It's time to test : ")
    #results = trainer.test(model, datamodule=dm, ckpt_path=best_link)

    predictions = trainer.predict(model, datamodule=dm, ckpt_path=best_link)
    validation, test, heldout = predictions


    dfs = []

    for split,str_split in zip(predictions,["val","test","heldout"]):
        original_pred = split

        split = np.concatenate(split)
        split = np.argmax(split, axis=1)
        y_split = pd.read_csv(config.data.dataset_path+"/"+str_split+".csv", low_memory=False)
        y_split = y_split[y_split["duration(s)"] < 30]
        y_split = y_split[ y_split["duration(s)"] > 0.04]

        
        df = y_split
        y_split = y_split["binary_classes"].to_numpy()

        split = np.array([ id2label[v] for v in split])


        df["add_predictions"] = split
        df.to_csv(save_path / ("results_" + str_split + ".csv"), index=False)


        print("y_split : ", y_split)
        print("split :", split)
        res = classification_report(y_split, split, output_dict=True)
        df = pd.DataFrame(res)
        df["model"] = args.run_id
        df["split"] = str_split
        df = df.reset_index()
        dfs += [df]


        cm = confusion_matrix(y_split, split, labels=["ADS", "KCDS", "Other"])
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ADS", "KCDS", "Other"]).plot()
        plt.savefig(save_path / ("CM_" + str_split + ".png"))

        plt.close()



    bigdf = pd.concat(dfs, ignore_index=True)
    bigdf.to_csv(save_path / "results.csv", index=False)
    print(bigdf)