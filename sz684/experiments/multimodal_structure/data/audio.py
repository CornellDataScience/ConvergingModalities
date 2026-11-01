"""Load real waveform samples; padding is never counted as observed sound."""
import numpy as np
from scipy.signal import resample_poly
from math import gcd


def load_audio(path, sample_rate=16000, max_seconds=10):
    """Decode with soundfile, average channels, resample, and trim (no padding)."""
    try:
        import soundfile as sf
        waveform, original_rate = sf.read(path, dtype='float32', always_2d=True)
        waveform = waveform.mean(axis=1)
    except ImportError:
        # PCM WAV pilot data can be read without installing optional codecs.
        from scipy.io import wavfile
        original_rate, waveform = wavfile.read(path)
        if np.issubdtype(waveform.dtype, np.integer):
            limit = max(abs(np.iinfo(waveform.dtype).min), np.iinfo(waveform.dtype).max)
            waveform = waveform.astype(np.float32) / limit
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError('Empty or nonfinite audio: ' + str(path))
    if sample_rate != original_rate:
        divisor = gcd(sample_rate, original_rate)
        waveform = resample_poly(waveform, sample_rate//divisor, original_rate//divisor)
    waveform = waveform[:int(sample_rate*max_seconds)]
    if np.max(np.abs(waveform)) < 1e-8:
        raise ValueError('Silent audio: ' + str(path))
    return waveform.astype(np.float32)
