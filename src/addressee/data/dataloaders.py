from pathlib import Path
from typing import Callable, List, Tuple, Union

from torch import Tensor
import lightning as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
#from addressee.utils.io import get_samples_in_range_seconds, get_audio_info


binary_classes = {"ADS":0,
                  "KCDS":1,
                  "Other":2}

ternary_classes = {"ADS":0,
                  "KCDS":1,
                  "OCDS":2,
                  "Other":3
                  }



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
            batch_size=self.config.train.batch_size,
            multiprocessing_context="fork"
            if torch.backends.mps.is_available()
            else None,
            ),
            DataLoader(
            heldout,
            num_workers=self.n_workers,
            pin_memory=True,
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
        self.exp_dir = Path(exp_dir)
        self.df = pd.read_csv(self.exp_dir /  (subset+".csv"), low_memory=False)
        self.df = self.df[self.df["duration(s)"] < 30]
        self.f_list, self.wav_onset, self.wav_offset, self.mask_onset, self.mask_offset, self.ind_list = self._get_lists(dataset, subset)
        #self.f_list, self.wav_archive, self.wav_bo, self.wav_bs, self.len_list, self.ind_list = self._get_lists(dataset, subset)
        #_LG.info(f"Finished loading wavs {subset} {len(self.f_list), len(self.ind_list), max(self.ind_list)}")
#        self.f_label, self.label_archive, self.label_bo, self.label_bs = self._load_labels(dataset, subset)

        self.mask_onset_index = (((self.mask_onset - self.wav_onset) * 16) / 320 - 1).astype(int) # modify scale
        self.mask_offset_index = (((self.mask_offset - self.wav_onset) * 16) / 320 + 1).astype(int)
        self.mask_offset_index[np.where(self.mask_offset_index > 1499) ] = 1499

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
        return self.df["file_path"].to_numpy(), self.df["filled_onset"].to_numpy(), self.df["filled_offset"].to_numpy(), self.df["segment_onset"].to_numpy(),self.df["segment_offset"].to_numpy(), np.asarray(list(range(len(self.df["file_path"]))))

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
            num_frames=480000,
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
        assert waveform.shape[1] == 480000 #30 secs
        return waveform.squeeze(1)

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
        #length = waveform.shape[1]
        start,end = (self.mask_onset_index[index], self.mask_offset_index[index])
        #print(self.f_label[index],self.f_list[index])
        #assert self.f_label[index] == self.f_list[index]
        t = torch.arange(1499)  # (1, T)

        mask = (t >= start) & (t < end)  # (B, T)
        mask = mask.unsqueeze(-1).float()   
        label = self.label_to_id[self.f_label[index]]   
        
        #label = torch.load(read_from_archive(self.f_label[index], self.label_bo[index], self.label_bs[index], self.label_archive_descriptors[self.label_archive[index]]))
        return (waveform, label, mask)
