"""Per-frame audio features: chroma (12-dim), RMS volume, onset envelope, tempo.

Audio side of the protocol from the 2021 Medium post, made explicit: the original
used a chromagram to weight ImageNet classes and volume/tempo to drive the noise
vector. Everything is extracted once into plain arrays so the mapping layer
(mapping.py) can be varied independently of the audio features.
"""
import numpy as np
import librosa
from scipy.interpolate import interp1d


def _interp_to(t_query, t_feat, x):
    axis = 1 if x.ndim == 2 else 0
    first = x[:, 0] if x.ndim == 2 else x[0]
    last = x[:, -1] if x.ndim == 2 else x[-1]
    f = interp1d(t_feat, x, axis=axis, bounds_error=False,
                 fill_value=(first, last))
    return f(t_query)


def _normalize(x, lo_pct=5, hi_pct=99):
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def extract(path, fps, seconds, start, sr=22050, hop=512):
    """Return dict of per-frame features at generation timestamps."""
    y, _ = librosa.load(path, sr=sr, mono=True, offset=start, duration=seconds + 2)
    times = np.arange(0, seconds, 1.0 / fps)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)   # (12, T)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]                  # (T,)
    tempo_values = librosa.feature.tempo(y=y, sr=sr, hop_length=hop)
    tempo = float(np.ravel(tempo_values)[0])
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)

    t_feat = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)
    C = _interp_to(times, t_feat, chroma).T    # (F, 12)
    R = _interp_to(times, t_feat, rms)         # (F,)
    B = _interp_to(times, t_feat, onset_env)   # (F,)

    C = C / (C.sum(1, keepdims=True) + 1e-9)
    return {"times": times, "chroma": C, "rms": _normalize(R),
            "beat": _normalize(B), "tempo": tempo}
