from pathlib import Path
from typing import List, Tuple, Union

from torch import Tensor
import lightning as pl
import numpy as np
import pandas as pd
from pandas import DataFrame
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio

binary_classes = {"ADS":0,
                  "KCDS":1,
                  "Other":2}

ternary_classes = {"ADS":0,
                  "KCDS":1,
                  "OCDS":2,
                  "Other":3
                  }

def extend_utterances(df, t):
    assert t >= 0
    df["fill"] = ((t - df["duration(s)"])/2).clip(lower=0)*1000
    onset_str = "filled_onset_" + str(t)
    offset_str = "filled_offset_" + str(t)
    df[onset_str] = df["segment_onset"] - df["fill"]
    df[offset_str] = df["segment_offset"] + df["fill"]
    df.loc[df[onset_str] < 0, offset_str] = t * 1000
    df.loc[df[onset_str] < 0, onset_str] = 0.0
    df.loc[df[offset_str] > df["recording_duration"], onset_str] = df["recording_duration"] - (t*1000)
    df.loc[df[offset_str] > df["recording_duration"], offset_str] = df["recording_duration"]
    
    return df

class AddresseeDataloader(pl.LightningDataModule):

    def __init__(
        self,
        dataset: str,
        dataset_path: Path,
        config,
        inference=False,
        optional_df = DataFrame | None,
        num_cpus=11
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.config = config
        self.inference = inference
        self.num_cpus = num_cpus
        self.df = optional_df
        if self.num_cpus > 11:
            self.n_workers = 16
        else:
            self.n_workers = 8
        self.rng = np.random.default_rng()

    def train_dataloader(self) -> DataLoader:
        dataset = AddresseeDataset(self.dataset_path, "addressee", "train", self.config)
        return DataLoader(
                dataset,
                batch_size=self.config.train.batch_size,
                drop_last=True,
                collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
                num_workers=self.n_workers,
                pin_memory=True,
                shuffle=True,
                persistent_workers=True,
                multiprocessing_context="fork"
                if torch.backends.mps.is_available()
                else None,
                )

    def val_dataloader(self) -> DataLoader:
        dataset = AddresseeDataset(self.dataset_path, "addressee", "val", self.config)
        return DataLoader(
            dataset,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size,
            persistent_workers=True,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            )
    
    def test_dataloader(self) -> DataLoader:
        test = AddresseeDataset(self.dataset_path, "addressee", "test", self.config)
        heldout = AddresseeDataset(self.dataset_path, "addressee", "heldout", self.config)
        return [DataLoader(
            test,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            ),
            DataLoader(
            heldout,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            )]
        
    def predict_dataloader(self) -> DataLoader:
        if self.inference:
            inference = AddresseeDataset(self.dataset_path, "addressee", "val", self.config, df=self.df)
            return DataLoader(
                inference,
                num_workers=self.n_workers,
                pin_memory=True,
                collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
                batch_size=self.config.train.batch_size,
                multiprocessing_context="fork"
                if torch.backends.mps.is_available()
                else None,
                )
        else:
            validation = AddresseeDataset(self.dataset_path, "addressee", "val", self.config)
            test = AddresseeDataset(self.dataset_path, "addressee", "test", self.config)
            heldout = AddresseeDataset(self.dataset_path, "addressee", "heldout", self.config)
            return [DataLoader(
                validation,
                num_workers=self.n_workers,
                pin_memory=True,
                collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
                batch_size=self.config.train.batch_size*4,
                multiprocessing_context="fork"
                if torch.backends.mps.is_available()
                else None,
                ),
                DataLoader(
                test,
                num_workers=self.n_workers,
                pin_memory=True,
                collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
                batch_size=self.config.train.batch_size*4,
                multiprocessing_context="fork"
                if torch.backends.mps.is_available()
                else None,
                ),
                DataLoader(
                heldout,
                num_workers=self.n_workers,
                pin_memory=True,
                collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=float(self.config.context_size) > 0),
                batch_size=self.config.train.batch_size*4,
                multiprocessing_context="fork"
                if torch.backends.mps.is_available()
                else None,
                )]

class AddresseeDataset(Dataset):
    """PyTorch Dataset for addressee detection model training and fine-tuning.

    Loads audio segments from CSV manifests, optionally extending them with
    surrounding context, and computes frame-level mask indices that indicate
    the span of the original utterance within the (possibly padded) context
    window fed to the model.

    Args:
        exp_dir (str or Path): Root directory containing the CSV split files
            (e.g. ``train.csv``, ``val.csv``, ``test.csv``).
        dataset (str): Dataset identifier (e.g. ``"addressee"``). Passed
            through to label-loading helpers.
        subset (str): Which split to load. Options: ``"train"``, ``"val"``,
            ``"test"``, ``"heldout"``.
        config: Experiment configuration object. Expected attributes include
            ``config.train.pad``, ``config.context_size``,
            ``config.data.classes``.
    """

    def __init__(
        self,
        exp_dir: Union[str, Path],
        dataset: str,
        subset: str,
        config,
        df = None,
    ) -> None:
        
        self.config = config
        self.pad = config.train.pad
        self.stride = 320
        self.context_size = float(self.config.context_size)   
        self.exp_dir = Path(exp_dir)

        if df is not None:
            self.df = df
        else:
            self.df = pd.read_csv(self.exp_dir /  (subset+".csv"), low_memory=False)
            self.df = self.df[self.df["duration(s)"] < 30]
            print(len(self.df))
            self.df = self.df[self.df["duration(s)"] > 0.04]
            print("dropped to :", len(self.df))

        self.df = extend_utterances(self.df, self.context_size)    
        self.f_list, self.wav_onset, self.wav_offset, self.mask_onset, self.mask_offset, self.ind_list = self._get_lists(dataset, subset)

        

        # Frame index layout for a context window fed to the model:
        #
        #   [padding_left | context_left | utterance | context_right | padding_right]
        #
        # Terminology:
        #   utterance_onset  / utterance_offset  – start/end of the utterance in
        #                                          the original audio file (ms)
        #   context_onset    / context_offset    – start/end of the audio segment
        #                                          actually fed to the model (ms)
        #
        # Frame index conversion (HuBERT stride = 320 samples @ 16 kHz):
        #   frame_index = ((sample_offset - 400) // stride)
        #
        #   context_onset_index  – last frame of the left context / first frame
        #                          of the utterance, clipped to 0
        #   context_offset_index – first frame of the right context / last frame
        #                          of the utterance

        self.utterance_onset = self.wav_onset
        self.utterance_offset = self.wav_offset
        
        self.context_onset = self.mask_onset
        self.context_offset = self.mask_offset

        self.context_onset_index = (((((self.utterance_onset - self.context_onset) * 16) - 400) // self.stride) - 2).astype(int)
        self.context_onset_index = self.context_onset_index.clip(0) 
        
        self.context_offset_index = ((((self.utterance_offset - self.context_onset) * 16) - 400) // self.stride ).astype(int)


        self.f_label = self._load_labels(dataset, subset, config)
        if config.data.classes == "binary_classes":
            self.label_to_id = binary_classes
        if config.data.classes == "ternary_classes":
            self.label_to_id = ternary_classes
        

    def __len__(self):
        return len(self.f_list)

    def _get_lists(
        self,
        dataset: str,
        subset: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build array-based index structures from the loaded DataFrame.

        When ``context_size`` is 0 the raw segment onsets/offsets are used as
        the audio window boundaries; otherwise the pre-computed
        ``filled_onset_<t>`` / ``filled_offset_<t>`` columns are used.

        Args:
            dataset (str): Dataset identifier (unused directly, kept for
                interface consistency).
            subset (str): Split name (unused directly, kept for interface
                consistency).

        Returns:
            f_list (np.ndarray): File paths for each sample.
            wav_onset (np.ndarray): Start of the audio window to load (ms).
            wav_offset (np.ndarray): End of the audio window to load (ms).
            mask_onset (np.ndarray): Start of the original utterance within
                the audio file (ms).
            mask_offset (np.ndarray): End of the original utterance within
                the audio file (ms).
            ind_list (np.ndarray): Sequential integer index for each sample.
        """

        if self.context_size == 0:
            onsets, offsets = self.df["segment_onset"].to_numpy(),self.df["segment_offset"].to_numpy()
        else:
            onsets, offsets = self.df["filled_onset_"+str(self.context_size)].to_numpy(), self.df["filled_offset_"+str(self.context_size)].to_numpy()
        return self.df["file_path"].to_numpy(), onsets, offsets, self.df["segment_onset"].to_numpy(),self.df["segment_offset"].to_numpy(), np.asarray(list(range(len(self.df["file_path"]))))

    def _load_audio(self, index: int) -> Tensor:
        """Load the waveform for a given sample index.

        Reads only the relevant slice of the audio file defined by
        ``wav_onset`` and ``wav_offset`` (converted from milliseconds to
        samples at 16 kHz).

        Args:
            index (int): The sample index into the dataset.

        Returns:
            Tensor: 1-D waveform tensor of shape ``(num_samples,)``.
        """
        waveform, sr = torchaudio.load(
                uri=Path(self.f_list[index]),
                frame_offset=int(self.wav_onset[index])*16,
                num_frames=int(self.wav_offset[index]*16 - self.wav_onset[index]*16)
                )
        return waveform.squeeze(0)

    def _load_labels(self, dataset: str, subset: str, config) -> np.ndarray:
        """Load all class labels into a numpy array.

        Args:
            dataset (str): Dataset identifier (unused directly, kept for
                interface consistency).
            subset (str): Split name (unused directly, kept for interface
                consistency).
            config: Experiment configuration object. ``config.data.classes``
                selects which column of the DataFrame contains the labels.

        Returns:
            np.ndarray: String label array aligned with the DataFrame rows.
        """
        return self.df[config.data.classes].to_numpy()

    def __getitem__(self, index):
        waveform = self._load_audio(index)
        length = waveform.shape[0]
        label = self.label_to_id[self.f_label[index]]   
        return (waveform, label, length, self.context_onset_index[index], self.context_offset_index[index])

class CollateFnHubert:
    """Collate function for HuBERT-based addressee detection batches.

    Center-pads all waveforms in a mini-batch to the length of the longest
    sample, then builds a per-sample binary frame mask that marks only the
    frames belonging to the original utterance (excluding left/right context
    padding). The mask is used downstream to pool only over the utterance
    frames when computing the classification loss.

    Args:
        pad (bool): If ``True``, waveforms are zero-padded to the maximum
            length in the batch. Must be ``True``; the unpadded path is not
            fully implemented. (Default: ``False``)
        rand_crop (bool): If ``True``, the starting index is chosen randomly
            when cropping to a minimum length. Currently unused. (Default: ``True``)
        additional_context (bool): Whether the samples include surrounding
            context beyond the raw utterance boundaries. When ``True``, the
            mask indices delimit the utterance within the wider context window.
            (Default: ``False``)
    """

    def __init__(
        self,
        pad: bool = False,
        rand_crop: bool = True,
        additional_context: bool = False
    ) -> None:
        self.pad = pad
        self.additional_context = additional_context
        self.rand_crop = rand_crop
        self.stride = 320

    def __call__(self, batch: List[Tuple[Tensor, int, int, int, int]]) -> Tuple[Tensor, Tensor, Tensor]:
        """Collate a list of dataset samples into a padded batch.

        Args:
            batch: List of tuples ``(waveform, label, length,
                left_context_end, right_context_start)`` as returned by
                ``AddresseeDataset.__getitem__``.

        Returns:
            waveforms (Tensor): Center-padded waveforms of shape
                ``(batch, time)``.
            labels (Tensor): Class indices of shape ``(batch,)``.
            lengths (Tensor): Original (unpadded) waveform lengths of shape
                ``(batch,)``.
            masks (Tensor): Binary utterance masks of shape
                ``(batch, num_frames, 1)``, where 1 indicates frames that
                belong to the utterance and 0 indicates context or padding
                frames. If all frames would be masked out for a sample the
                mask is set to all-ones to avoid NaN losses.
        """
        waveforms, labels, lengths, left_context_end, right_context_start = [], [], [], [], []
        max_wav_length = 0

        for sample in batch:
            waveform, label, length, lce, rcs = sample

            if waveform.shape[0] > max_wav_length:
                max_wav_length = waveform.shape[0]

            waveforms.append(waveform)
            lengths.append(length)
            labels.append(label)
            left_context_end.append(lce)
            right_context_start.append(rcs)

        # make sure the shapes are the same if not apply zero-padding
        if not self.pad:
            assert all(
                [waveform.shape[0] == waveforms[0].shape[0] for waveform in waveforms]
            ), "The dimensions of the waveforms should be identical in the same batch."
            assert all(
                [label.shape[0] == labels[0].shape[0] for label in labels]
            ), "The dimensions of the labels should be identical in the same batch."


        
        masks = []
        # Center-pad each waveform and compute utterance frame masks.
        # Only left-pad size is needed for index adjustment since the right
        # context boundary is guaranteed to fall before the right padding.
        lengths = np.array(lengths)
        pad_left = np.floor((max_wav_length - lengths) / 2)
        pad_right = np.ceil((max_wav_length - lengths) / 2)
        padded_wavs = []
        for i,w in enumerate(waveforms):
            padded_wavs.append(torch.nn.functional.pad(w, (int(pad_left[i]), int(pad_right[i])), "constant",0))
    
            # Frame indices for the longest waveform in the batch
            t = torch.arange(int( ((max_wav_length - 400) // 320) + 1))

            # Mask is 1 for frames inside [left_context_end, right_context_start)
            # after accounting for the left zero-padding offset.
            mask = (t >= int(pad_left[i]/320) + left_context_end[i]) & (t < int(pad_left[i]/320) + right_context_start[i])
            mask = mask.float().unsqueeze(-1)
            
            # Guard against an all-zero mask to prevent NaN in the loss
            if sum(mask) == 0:
                mask = mask +1
            masks += [mask]

            assert ((padded_wavs[-1].shape[0] - 400) // 320 )+1 == mask.shape[0]

        waveforms = torch.stack(padded_wavs, dim=0)
        masks = torch.stack(masks, dim=0)
        labels = torch.tensor(labels)

        return waveforms, labels, masks # start_masks, end_masks #attn_mask, mask_indices #