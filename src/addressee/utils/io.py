from dataclasses import dataclass
from pathlib import Path

import torch
#from torchcodec.decoders import AudioDecoder
#from torchcodec.encoders import AudioEncoder

@dataclass
class AudioInfo:
    sample_rate: int
    n_samples: int
    n_channels: int


# def get_audio_info(audio_p: Path) -> AudioInfo:
#     """Returns number of frames and sample rate of the audio"""
#     decoder = AudioDecoder(audio_p.resolve())
#     return AudioInfo(
#         n_samples=int(
#             decoder.metadata.duration_seconds_from_header * decoder.metadata.sample_rate
#         ),
#         sample_rate=decoder.metadata.sample_rate,
#         n_channels=decoder.metadata.num_channels,
#     )





# def get_samples_in_range_seconds(audio_p: Path, start_s: int, duration_s: int) -> torch.Tensor:
#     """Get samples in range [start_f : start_f + duration_f].

#     Args:
#         audio_p (Path): Path to the audio file in `.wav` format.
#         start_f (int): start frame to start loading the audio samples.
#         duration_f (int): duration of the loaded window. If `duration_f==-1` the the rest of the audio is loaded

#     Returns:
#         torch.Tensor: Tensor containing the loaded data, with shape (n_channels, n_samples).
#     """
#     decoder = AudioDecoder(audio_p.resolve())
#     return decoder.get_samples_played_in_range(
#         start_seconds=start_s,
#         stop_seconds=None
#         if duration_s < 0
#         else start_s + duration_s,
#     ).data



# def get_all_samples(audio_p: Path) -> torch.Tensor:
#     decoder = AudioDecoder(audio_p.resolve())
#     return decoder.get_all_samples().data


# def write_data_to_disk(
#     data: torch.Tensor, output_file: Path, sample_rate: int = 16_000
# ) -> None:
#     AudioEncoder(data, sample_rate=sample_rate).to_file(output_file.with_suffix(".wav"))