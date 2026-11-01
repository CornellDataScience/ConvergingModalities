"""Audio-only AudioMAE pretraining backbone, with its published processor.

The converted timm weights have no supervised classification head. Weight hashes
and conversion provenance are retained in the model specification.
"""
import torch
import torch.nn.functional as functional
from ..data.audio import load_audio


def load_model(spec, device):
    """Use the declared spectrogram size, without downloading or fine-tuning."""
    import timm
    from safetensors.torch import load_file
    model = timm.create_model('vit_base_patch16_224', pretrained=False,
                             img_size=(1024,128), in_chans=1, num_classes=0)
    model.load_state_dict(load_file(spec['weights_path']), strict=True)
    return model.eval().to(device)


def spectrogram(waveform):
    """Published AudioMAE input: 128 filterbanks, 1024 frames, fixed statistics.

    Pad in filterbank space before normalization, matching the published recipe.
    This fixed padded window is part of AudioMAE's input, not extra recordings.
    """
    from torchaudio.compliance import kaldi
    audio = torch.tensor(waveform[None,:])
    banks = kaldi.fbank(audio, htk_compat=True, window_type='hanning', num_mel_bins=128)
    banks = banks[:1024]
    if len(banks)<1024:
        banks = functional.pad(banks,(0,0,0,1024-len(banks)))
    return ((banks + 4.2677393)/(4.5689974*2))[None,:,:]


def extract_batch(records, model, spec, device):
    """Pool the native CLS token; all models retain their own processor."""
    inputs = [spectrogram(load_audio(row['path'],16000,10)) for row in records]
    with torch.inference_mode():
        tokens = model.forward_features(torch.stack(inputs).to(device))
    return tokens[:,0].float().cpu().numpy()
