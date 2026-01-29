from __future__ import annotations
import numpy as np

def _find_first_arrival(h: np.ndarray) -> int:
    return int(np.argmax(np.abs(h)))

def rt60_schroeder(h: np.ndarray, sr: int) -> float:
    h = np.asarray(h, dtype=np.float64).squeeze()
    e = h**2
    edc = np.flip(np.cumsum(np.flip(e)))
    if edc[0] <= 0:
        return float("nan")
    edc_db = 10.0 * np.log10(np.maximum(edc / edc[0], 1e-20))
    lo, hi = -35.0, -5.0
    idx = np.where((edc_db <= hi) & (edc_db >= lo))[0]
    if len(idx) < 10:
        return float("nan")
    t = idx / float(sr)
    y = edc_db[idx]
    A = np.vstack([t, np.ones_like(t)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if a >= 0:
        return float("nan")
    t60 = (-60.0 - b) / a
    return float(max(t60, 0.0))

def drr_direct_window(h: np.ndarray, sr: int, window_ms: float = 2.5) -> float:
    h = np.asarray(h, dtype=np.float64).squeeze()
    n = len(h)
    peak = _find_first_arrival(h)
    win = int(round(window_ms * 1e-3 * sr))
    win = max(win, 1)
    half = win // 2
    s = max(0, peak - half)
    e = min(n, s + win)
    direct = h[s:e]
    mask = np.ones(n, dtype=bool)
    mask[s:e] = False
    rev = h[mask]
    ed = np.sum(direct**2)
    er = np.sum(rev**2)
    if er <= 0:
        return float("nan")
    return float(10.0 * np.log10(max(ed, 1e-20) / er))

def c50_from_rir(h: np.ndarray, sr: int) -> float:
    h = np.asarray(h, dtype=np.float64).squeeze()
    n50 = int(round(0.050 * sr))
    n50 = min(max(n50, 1), len(h))
    early = h[:n50]
    late = h[n50:]
    ee = np.sum(early**2)
    el = np.sum(late**2)
    if el <= 0:
        return float("nan")
    return float(10.0 * np.log10(max(ee, 1e-20) / el))
