from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf

def load_audio_mono(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return x.astype(np.float32), sr

def write_wav_int16(path: Path, x: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    if peak > 0.999:
        x = x / peak * 0.999
    sf.write(str(path), x, sr, subtype="PCM_16")

def convolve_time(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    import scipy.signal
    return scipy.signal.fftconvolve(x, h, mode="full").astype(np.float32)
