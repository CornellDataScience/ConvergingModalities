"""Real PCM WAV decoding/resampling checks, without model downloads."""
import numpy as np
import pytest
from scipy.io import wavfile
from ..data.audio import load_audio


def test_audio_downmix_resample_and_trim(tmp_path):
    rate = 8000
    time = np.arange(rate*2)/rate
    wave = np.sin(2*np.pi*440*time)*0.4
    path = tmp_path/'stereo.wav'
    wavfile.write(path, rate, np.column_stack([wave,wave]).astype(np.float32))
    actual = load_audio(path, sample_rate=16000, max_seconds=1)
    assert actual.shape == (16000,)
    assert np.max(np.abs(actual)) == pytest.approx(0.4, abs=0.01)
    assert np.isfinite(actual).all()


def test_silent_audio_rejected(tmp_path):
    path = tmp_path/'silence.wav'
    wavfile.write(path,16000,np.zeros(16000,dtype=np.int16))
    with pytest.raises(ValueError, match='Silent'):
        load_audio(path)
