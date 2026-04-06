
from addressee.data.dataloaders import extend_utterances, CollateFnHubert
from addressee.models.hubert.modeling_hubert import HubertFinetune
from addressee.utils.config import load_config
import torch
import pandas as pd
import numpy as np

# def test_extend_utterances_negative():

#     df = pd.DataFrame()
#     df = extend_utterances(df)
#     assert df["fill"] > 0 


# def test_masking():
#     cfg = load_yaml("/store/scratch/tcharlot/addressee_models/V3/bbh2Context10Sseed4/checkpoints/epoch=02-val_loss=0.650.ckpt")
#     model = HubertFinetune(cfg)

#     collator = CollateFnHubert(pad=True, rand_crop=False, additional_context=10.0)

#     for i in range(2000):
#         batch = torch.rand((16,i))
#         labels = torch.tensor(np.random.randint(3, size=16))
#         lengths = torch.ones((16,i)) * i
#         lces = torch.zeros(16)
#         rces = torch.zeros(16)

#         batch = [(batch[j], labels[j], lengths[j], lces[j], rces[j]) for j in range(16)]
#         batch = collator(batch)

#         model.forward(batch[0], batch[-1])



def test_masking():
    path_model = "/store/scratch/tcharlot/addressee_models/V3/bbh2Context10Sseed4/checkpoints/epoch=02-val_loss=0.650.ckpt"
    cfg = load_config(train_config="/store/scratch/tcharlot/addressee_models/V3/bbh2Context10Sseed4/config.yaml")
    #print(cfg.keys())
    cfg.model_checkpoint = path_model
    model = HubertFinetune(cfg)
    model.eval()
    collator = CollateFnHubert(pad=True, rand_crop=False, additional_context=True)
    batch_size = 4
    STRIDE = 320

    for wav_length in range(30000, 32000):
        # Vary utterance bounds within the waveform to stress-test masking index computation
        utt_onset = wav_length // 4
        utt_offset = 3 * wav_length // 4
        # Ensure utterance has nonzero length
        if utt_offset <= utt_onset:
            utt_offset = utt_onset + 1

        # Compute context onset/offset indices the same way AddresseeDataset does
        # context_onset_index = (((utt_onset * 16) - 400) // STRIDE - 2).clip(0)
        # context_offset_index = (((utt_offset * 16) - 400) // STRIDE)
        lce = max(0, int((((utt_onset) - 400) // STRIDE) - 2))
        rcs = int(((utt_offset) - 400) // STRIDE)

        batch = [
            (
                torch.rand(wav_length),               # waveform of length wav_length
                int(np.random.randint(3)),             # label in {0,1,2}
                wav_length,                            # length
                lce,                                   # left context end index (HuBERT frames)
                rcs,                                   # right context start index (HuBERT frames)
            )
            for _ in range(batch_size)
        ]

        waveforms, labels, lengths, masks = collator(batch)

        # Basic shape/type assertions before forward
        assert waveforms.ndim == 2, f"Expected 2D waveforms, got {waveforms.shape}"
        assert masks.ndim == 3, f"Expected 3D masks, got {masks.shape}"
        n_frames = ((waveforms.shape[1] - 400) // STRIDE) + 1
        assert masks.shape == (batch_size, n_frames, 1), (
            f"Mask shape mismatch: got {masks.shape}, expected (16, {n_frames}, 1) "
            f"for wav_length={wav_length}"
        )
        # Mask must not be all-zero (collator sets it to 1 in that case, but let's verify)
        assert masks.sum() > 0, f"Mask is all zero for wav_length={wav_length}"

        with torch.no_grad():
            logits = model.forward(waveforms, lengths=None, mask=masks)

        assert logits.shape == (batch_size, 3), (
            f"Unexpected logits shape {logits.shape} for wav_length={wav_length}"
        )