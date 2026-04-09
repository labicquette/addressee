import argparse
import copy
import logging
import shutil
from pathlib import Path
from typing import Literal
import torch
import os
from glob import glob 
from huggingface_hub import hf_hub_download
from torchcodec.decoders import AudioDecoder
from tqdm import tqdm
import pandas as pd 
import lightning as pl
from addressee.data.dataloaders import AddresseeDataloader
from addressee.utils.config import load_config
from addressee.models.hubert.modeling_hubert import HubertFinetune
import numpy as np

id2label = {0: "ADS",
                  1: "KCDS",
                  2: "Other"}



def process_inference_outputs(predictions, df):
    predictions = np.concatenate(predictions)
    predictions = np.argmax(predictions, axis=1)
    predictions = np.array([ id2label[v] for v in predictions])
    df.loc[(df["label"] == "FEM") | (df["label"] == "MAL"), "addressee"] = predictions
    df = df.drop(columns=["binary_classes", "segment_onset", "segment_offset", "duration(s)", "file_path", "recording_duration"])

    return df

def main(
    output: str,
    uris: Path | None = None,
    VTC2_output: Path | None = None,
    wavs: str = "data/debug/wav",
    save_logits: bool = False,
    batch_size: int = 128,
    context_size : float = 10.0,
    write_empty: bool = True,
    write_csv: bool = True,
    device: Literal["gpu", "cuda", "cpu", "mps"] = "gpu",
    model_dir: str | None = None,
    model_revision: str | None = None,
    #keep_raw: bool = False,
    verbose=True,
):
    
    model_path = hf_hub_download(repo_id="coml/addressee", filename="best.ckpt", local_dir=model_dir, revision = model_revision)
    config_path = hf_hub_download(repo_id="coml/addressee", filename="config.yaml", local_dir=model_dir, revision = model_revision)

    config = load_config(train_config=config_path)
    config.train.batch_size = 32
    config.model_id = model_path
    
    model = HubertFinetune(config)
    model.eval()
    
    #filter abnormal audios ?
    df_VTC2 = pd.read_csv(VTC2_output, low_memory=False)



    df_VTC2["recording_duration"] = -1.0
    df_VTC2["file_path"] = ""
    for uid in tqdm(df_VTC2["uid"].unique()):
        wav_file = wavs + "/"+ uid + ".wav"
        decoder = AudioDecoder(wav_file)
        duration = decoder.metadata.duration_seconds
        df_VTC2.loc[df_VTC2["uid"] == uid, "recording_duration"] = duration
        df_VTC2.loc[df_VTC2["uid"] == uid, "file_path"] = wav_file
    
    #segment_onset in milliseconds
    df_VTC2["segment_onset"] = df_VTC2["start_time_s"] * 1000
    df_VTC2["segment_offset"] = df_VTC2["segment_onset"] + (df_VTC2["duration_s"] * 1000)
    df_VTC2["duration(s)"] =df_VTC2["duration_s"]
    df_VTC2["addressee"] = pd.NA
    df_VTC2["binary_classes"] = "KCDS"

    df_inference = df_VTC2.loc[(df_VTC2["label"] == "FEM") | (df_VTC2["label"] == "MAL")].copy()

    dm = AddresseeDataloader(dataset = "addressee",
                             dataset_path= config.data.dataset_path,
                             inference=True,
                             optional_df=df_inference,
                             config= config)



    trainer = pl.Trainer(
        accelerator=device,
        devices=1,
        logger=False
    )

    predictions = trainer.predict(model, datamodule=dm, ckpt_path=model_path)

    df_VTC2 = process_inference_outputs(predictions, df_VTC2)

    df_VTC2.to_csv(output+"/rttm_addressee.csv", index=False)
    return 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uris", help="Path to a file containing the list of uris to use."
    )
    parser.add_argument(
        "--VTC2_output", help="Path to a file containing the list of uris to use."
    )
    parser.add_argument(
        "--wavs",
        default="data/debug/wav",
        help="Folder containing the audio files to run inference on.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output Path to the folder that will contain the final predictions.",
    )
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="If the prediction scripts saves the logits to disk, can be memory intensive.",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="Batch size to use for the forward pass of the model.",
    )
    parser.add_argument(
        "--context_size",
        default=10.0,
        type=float,
        help="Context window to use for the forward pass of the model.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["gpu", "cuda", "cpu", "mps"],
        help="Size of the batch used for the forward pass in the model.",
    )
    torch.set_float32_matmul_precision('high')
    args = parser.parse_args()
    main(**vars(args))