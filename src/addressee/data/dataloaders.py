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
        if int(self.config.context_size) > 0:
            self.static_mask = int(((int(self.config.context_size) * 16000) / 320) - 1) 
        else:
            self.static_mask = 0
            
        self.exp_dir = Path(exp_dir)
        self.df = pd.read_csv(self.exp_dir /  (subset+".csv"), low_memory=False)

        self.context_size = int(self.config.context_size)        
        
        self.df = self.df[self.df["duration(s)"] < 30]
        print(len(self.df))
        self.df = self.df[self.df["duration(s)"] > 0.04]
        print("dropped to :", len(self.df))
        self.f_list, self.wav_onset, self.wav_offset, self.mask_onset, self.mask_offset, self.ind_list = self._get_lists(dataset, subset)


        
        self.mask_onset_index = (((self.mask_onset - self.wav_onset) * 16) / self.stride - 1).astype(int) # modify scale
        self.mask_onset_index = self.mask_onset_index.clip(0) 
        if self.context_size > 0:
            self.mask_offset_index = (((self.mask_offset - self.wav_onset) * 16) / self.stride + 1).astype(int)
            self.mask_offset_index[np.where(self.mask_offset_index > self.static_mask) ] = self.static_mask
        else:
            self.mask_offset_index = (((self.wav_offset - self.wav_onset) * 16) / self.stride + 1).astype(int)
            self.mask_offset_index[np.where(self.mask_offset_index > self.static_mask) ] = self.static_mask
        #print(self.mask_onset_index, self.mask_offset_index)

        self.f_label = self._load_labels(dataset, subset, config)
        if config.data.classes == "binary_classes":
            self.label_to_id = binary_classes
        if config.data.classes == "ternary_classes":
            self.label_to_id = ternary_classes
        #_LG.info(f"Finished loading dataset {subset}")
        

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

        #self.wav_archive_descriptors = {f:open(f, "rb") for f in self.df["archive_wav"].unique()}
        # if self.pad:
        #     onsets, offsets = self.df["segment_onset"].to_numpy(), self.df["segment_offset"].to_numpy() 
        # else:
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
        #print(self.wav_onset[index], self.wav_offset[index])
        # waveform, _sr = torchaudio.load(
        #     Path(self.f_list[index]),
        #     int(self.wav_onset[index]),
        #     num_frames=480000
        # )
        waveform, sr = torchaudio.load(
                uri=Path(self.f_list[index]),
                frame_offset=int(self.wav_onset[index])*16,
                #check if its ms or frames for segment_onset vs filled_onset
                num_frames=int(self.wav_offset[index]*16 - self.wav_onset[index]*16),
                backend="soundfile"
                )
        # waveform = get_samples_in_range_seconds(
        #         Path(self.f_list[index]),
        #         int(self.wav_onset[index])/1000,# onsets are in milliseconds
        #         480000
        #         )
        # except:
        #     print("problem : ", self.f_list[index], get_audio_info(Path(self.f_list[index])), self.wav_onset[index]/16000, (self.wav_onset[index]/16000) + 30)
        #     waveform = get_samples_in_range_seconds(
        #         Path(self.f_list[index]),
        #         int(self.wav_onset[index])/16000,
        #         30
        #         )
        # #waveform = get_samples_in_range_seconds(, self.wav_onset[index], self.wav_offset[index])
        #waveform = torch.load(read_from_archive(self.f_list[index], self.wav_bo[index], self.wav_bs[index], self.wav_archive_descriptors[self.wav_archive[index]]))        
        #assert waveform.shape[1] == 480000 #30 secs
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
        #_LG.info(f"Loading labels {subset}")
        #_LG.info(f"Generating descriptors {subset}")
        #self.label_archive_descriptors = {f:open(f, "rb") for f in self.df["archive_lab"].unique()}
        return self.df[config.data.classes].to_numpy()
        #return self.df["path_lab"].to_numpy(), self.df["archive_lab"].to_numpy(), self.df["byte_offset_lab"].to_numpy(), self.df["byte_size_lab"].to_numpy()

    def __getitem__(self, index):
        waveform = self._load_audio(index)
        length = waveform.shape[0]
        #length = waveform.shape[1]
        if self.context_size > 0:
            start,end = (self.mask_onset_index[index], self.mask_offset_index[index])
            #print(self.f_label[index],self.f_list[index])
            #assert self.f_label[index] == self.f_list[index]
            t = torch.arange(int(length/320)-1)
            mask = (t >= start) & (t < end)  # (B, T)
            mask = mask.unsqueeze(-1).float()
        # else:
        #     #print(length)
        #     mask = torch.ones(int(length/320)-1)
        #inner_onset, inner_offset = self.mask_onset[index], self.mask_offset[index]
        label = self.label_to_id[self.f_label[index]]   
        #print("length of waveform : ", length)
        #label = torch.load(read_from_archive(self.f_label[index], self.label_bo[index], self.label_bs[index], self.label_archive_descriptors[self.label_archive[index]]))
        return (waveform, label, length, self.mask_onset_index[index], self.mask_offset_index[index])


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
        if self.pad:
            num_frames = max([sample[0].shape[0] for sample in batch])
        else:
            num_frames = min([sample[0].shape[1] for sample in batch])

        #num_frames taille du crop 
        # Logger: num_frames, wav_lengths, labels_lengths
        #_LG.info(f"min_frames : {num_frames} before : {[(b[0].shape, b[1].shape,b[2]) for b in batch]}")
        waveforms, labels, lengths, start_masks, end_masks = [], [], [], [], []
        for sample in batch:
            waveform, label, length, start_mask, end_mask = sample
            
            waveforms.append(waveform)
            lengths.append(length)
            labels.append(label)
            start_masks.append(start_mask)
            end_masks.append(end_mask)

        # make sure the shapes are the same if not apply zero-padding
        if not self.pad:
            #_LG.info(f"after : {[(waveforms[i].shape, labels[i].shape,lengths[i]) for i,b in enumerate(batch)]}")
            #_LG.info(f"labels list : {[(label.shape[0],labels[0].shape[0]) for label in labels]}")
            assert all(
                [waveform.shape[0] == waveforms[0].shape[0] for waveform in waveforms]
            ), "The dimensions of the waveforms should be identical in the same batch."
            assert all(
                [label.shape[0] == labels[0].shape[0] for label in labels]
            ), "The dimensions of the labels should be identical in the same batch."


        
        waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        
        
        lengths = torch.tensor(lengths)
        #hubert_lengths = ((lengths + self.stride - 1) // self.stride)
        labels = torch.tensor(labels)
        #masks = torch.tensor(masks)
        start_masks = torch.tensor(start_masks)
        end_masks = torch.tensor(end_masks)



        # lengths = conv_length(self.conv_layer_config, lengths)
        # batch_size, max_len = waveforms.size(0), int(lengths.max())
        # padding_mask = torch.arange(max_len, device=lengths.device).expand(batch_size, max_len) >= lengths[:, None]
        # attn_mask = ~padding_mask[:, None, None, :].expand(batch_size, 1, max_len, max_len)
        # mask_indices = self.mask_generator(padding_mask)[0]
        
        return waveforms, labels, lengths, start_masks, end_masks #attn_mask, mask_indices #