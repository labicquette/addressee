from pathlib import Path
from typing import Callable, List, Tuple, Union

from torch import Tensor
import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import math
#from addressee.utils.io import get_samples_in_range_seconds, get_audio_info


binary_classes = {"ADS":0,
                  "KCDS":1,
                  "Other":2}

ternary_classes = {"ADS":0,
                  "KCDS":1,
                  "OCDS":2,
                  "Other":3
                  }


# taken from spidr https://github.com/facebookresearch/spidr
# DEFAULT_CONV_LAYER_CONFIG: list[tuple[int, int, int]] = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2
# def conv_length(shapes: list[tuple[int, int, int]], length: Tensor) -> Tensor:
#     for _, kernel_size, stride in shapes:
#         length = torch.div(length - kernel_size, stride, rounding_mode="floor") + 1
#         length = torch.max(torch.zeros_like(length), length)
#     return length



class AddresseeDataloader(pl.LightningDataModule):

    def __init__(
        self,
        dataset: str,
        dataset_path: Path,
        config,
        testing_test="test",
        num_cpus=11
        #conv_settings: ConvolutionSettings,
        #audio_preparation_hook: Callable | None = None
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.config = config
        self.testing_set = testing_test
        #self.conv_settings = conv_settings
        #self.audio_preparation_hook = audio_preparation_hook
        self.num_cpus = num_cpus
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
                collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=int(self.config.context_size) > 0),
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
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=int(self.config.context_size) > 0),
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
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=int(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            ),
            DataLoader(
            heldout,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=int(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            )]
    def predict_dataloader(self) -> DataLoader:
        validation = AddresseeDataset(self.dataset_path, "addressee", "val", self.config)
        test = AddresseeDataset(self.dataset_path, "addressee", "test", self.config)
        heldout = AddresseeDataset(self.dataset_path, "addressee", "heldout", self.config)
        return [DataLoader(
            validation,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=int(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size*4,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            ),
            DataLoader(
            test,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=int(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size*4,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            ),
            DataLoader(
            heldout,
            num_workers=self.n_workers,
            pin_memory=True,
            collate_fn=CollateFnHubert(pad=True, rand_crop=False, additional_context=int(self.config.context_size) > 0),
            batch_size=self.config.train.batch_size*4,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            )]

class AddresseeDataset(Dataset):
    """Create a Dataset for HuBERT model training and fine-tuning.

    Args:
        exp_dir (str or Path): The root directory of the ``.tsv`` file list.
        dataset (str): The dataset for training. Options: [``librispeech``, ``librilight``].
        subset (str): The subset of the dataset. Options: [``train``, ``valid``].
    """

    def __init__(
        self,
        exp_dir: Union[str, Path],
        dataset: str,
        subset: str,
        config
    ) -> None:
        
        self.config = config
        self.pad = config.train.pad
        self.stride = 320

        self.exp_dir = Path(exp_dir)
        self.df = pd.read_csv(self.exp_dir /  (subset+".csv"), low_memory=False)

        self.context_size = int(self.config.context_size)        
        
        self.df = self.df[self.df["duration(s)"] < 30]
        print(len(self.df))
        self.df = self.df[self.df["duration(s)"] > 0.04]
        print("dropped to :", len(self.df))
        self.f_list, self.wav_onset, self.wav_offset, self.mask_onset, self.mask_offset, self.ind_list = self._get_lists(dataset, subset)


        # 1499 = 30s waveform
        # 0 : mask_onset
        # 100 : wav_onset
        # 800 : wav_offset
        # 1499 : mask_offset
        # [0:1499]
        # masking should be :  
        # [padding_left[context_left[wav_onset:wav_offset]context_right]padding_right]
        # from indices : 
        # mask_onset becomes the 0 index
        # wav_onset index = (wav_onset - mask_onset)/320 - 1
        # wav_offset index = (wav_offset - mask_onset)/320 - 1

        # utterance_onset = wav_onset (start of the utterance in the original audio file)
        # utterance_offset = wav_onset (end of the utterance in the original audio file)
        # context_onset = mask_onset (start of the audio segment feeded to the model) 
        # context_offset = mask_offset (end of the audio segment feeded to the model) 
        self.utterance_onset = self.wav_onset
        self.utterance_offset = self.wav_offset
        
        self.context_onset = self.mask_onset
        self.context_offset = self.mask_offset


        # index of the end of the left context AND start of utterance index
        self.context_onset_index = (((self.utterance_onset - self.context_onset) * 16) / self.stride - 1).astype(int) # modify scale
        self.context_onset_index = self.context_onset_index.clip(0) 
        
        # index of the start of the right context AND end of utterance index
        self.context_offset_index = (((self.utterance_offset - self.context_onset) * 16) / self.stride + 1).astype(int)
        


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
    ) -> Tuple[List[Path], List[int], List[int]]:
        """Get the list of paths for iteration.
        Args:
            tsv_dir (Path): The root directory of the ``.tsv`` file list.
            dataset (str): The dataset for training. Options: [``librispeech``, ``librilight``].
            subset (str): The subset of the dataset. Options: [``train``, ``valid``].

        Returns:
            (numpy.array) List of file paths.
            (numpy.array) List of indices.
            (numpy.array) List of waveform lengths.
        """

        if self.context_size == 0:
            onsets, offsets = self.df["segment_onset"].to_numpy(),self.df["segment_offset"].to_numpy()
        else:
            onsets, offsets = self.df["filled_onset_"+str(self.context_size)].to_numpy(), self.df["filled_offset_"+str(self.context_size)].to_numpy()
        return self.df["file_path"].to_numpy(), onsets, offsets, self.df["segment_onset"].to_numpy(),self.df["segment_offset"].to_numpy(), np.asarray(list(range(len(self.df["file_path"]))))

    def _load_audio(self, index: int) -> Tensor:
        """Load waveform given the sample index of the dataset.
        Args:
            index (int): The sample index.

        Returns:
            (Tensor): The corresponding waveform Tensor.
        """
        waveform, sr = torchaudio.load(
                uri=Path(self.f_list[index]),
                frame_offset=int(self.wav_onset[index])*16,
                #check if its ms or frames for segment_onset vs filled_onset
                num_frames=int(self.wav_offset[index]*16 - self.wav_onset[index]*16),
                backend="soundfile"
                )
        return waveform.squeeze(0)

    def _load_labels(self, dataset: str, subset: str, config) -> np.array:
        """Load all labels to memory into a numpy array.
        Args:
            label_dir (Path): The directory that contains the label file.
            dataset (str): The dataset for training. Options: [``librispeech``, ``librilight``].
            subset (str): The subset of the dataset. Options: [``train``, ``valid``].

        Returns:
            (np.array): The numpy arrary that contains the labels for each audio file.
        """
        return self.df[config.data.classes].to_numpy()

    def __getitem__(self, index):
        waveform = self._load_audio(index)
        length = waveform.shape[0]
        label = self.label_to_id[self.f_label[index]]   
        return (waveform, label, length, self.context_onset_index[index], self.context_offset_index[index])


def _get_padding_mask(input: Tensor, lengths: Tensor) -> Tensor:
    """Generate the padding mask given the padded input and the lengths Tensors.
    Args:
        input (Tensor): The padded Tensor of dimension `[batch, max_len, frequency]`.
        lengths (Tensor): The lengths Tensor of dimension `[batch,]`.

    Returns:
        (Tensor): The padding mask.
    """
    batch_size, max_len = input.shape
    mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
    return mask


class CollateFnHubert:
    """The collate class for HuBERT pre-training and fine-tuning.
    Args:
        feature_type (str): The type of features for KMeans clustering.
            Options: [``mfcc``, ``hubert``].
        pad (bool): If ``True``, the waveforms and labels will be padded to the
            max length in the mini-batch. If ``pad`` is False, the waveforms
            and labels will be cropped to the minimum length in the mini-batch.
            (Default: False)
        rand_crop (bool): if ``True``, the starting index of the waveform
            and label is random if the length is longer than the minimum
            length in the mini-batch.
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

    def __call__(self, batch: List[Tuple[Tensor, Tensor, int]]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            batch (List[Tuple(Tensor, Tensor, int)]):
                The list of tuples that contains the waveforms, labels, and audio lengths.

        Returns:
            (Tuple(Tensor, Tensor, Tensor)):
                The Tensor of waveforms with dimensions `(batch, time)`.
                The Tensor of labels with dimensions `(batch, seq)`.
                The Tensor of audio lengths with dimension `(batch,)`.
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
        # pad by hand, track left padding only because index based masking, right_context_start <= pad_right_start
        lengths = np.array(lengths)
        pad_left = np.floor((max_wav_length - lengths) / 2)
        pad_right = np.ceil((max_wav_length - lengths) / 2)
        padded_wavs = []
        for i,w in enumerate(waveforms):
            padded_wavs.append(torch.nn.functional.pad(w, (int(pad_left[i]), int(pad_right[i])), "constant",0))
    

            # make indices of frames for longest waveform in batch
            t = torch.arange(int((max_wav_length/320))-1)

            # make mask based on :
            # [0 : left padding + left_context_end] and [left padding + start of right context:] gives mask to compute pooling only on original utterances
            mask = (t >= int(pad_left[i]/320) + left_context_end[i]) & (t < int(pad_left[i]/320) + right_context_start[i])
            mask = mask.float().unsqueeze(-1)
            
            #case where mask == 0 not possible then mask to 1; if not nans in loss 
            if sum(mask) == 0:
                mask = mask +1
            masks += [mask]

        waveforms = torch.stack(padded_wavs, dim=0)
        masks = torch.stack(masks, dim=0)
        lengths = torch.tensor(lengths)
        labels = torch.tensor(labels)

        return waveforms, labels, lengths, masks # start_masks, end_masks #attn_mask, mask_indices #