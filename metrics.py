"""Evaluation metrics for the generation matrix.

Objective metrics
  * LPIPS          : perceptual distance between consecutive frames (flicker)
                     and between random frame pairs (perceptual drift).
  * Temporal diff  : mean |I_t - I_{t-1}| (pixel-level motion magnitude).
  * Motion sync    : correlation between visible frame-to-frame change and the
                     shared motion request. This is evaluated against shifted,
                     shuffled, flat, and distribution-matched random controls.
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr


_LPIPS_MODEL = None


def load_frames(out_dir):
    import glob, imageio
    files = sorted(glob.glob(f"{out_dir}/frames/*.png"))
    return np.stack([imageio.imread(f).astype(np.float32) / 255.0 for f in files])


def load_meta(out_dir):
    return np.load(f"{out_dir}/meta.npz", allow_pickle=True)


def temporal_diff(frames):
    """Mean absolute pixel change between consecutive RGB frames."""
    motion = visual_motion_signal(frames)
    return float(motion.mean()) if len(motion) else float("nan")


def visual_motion_signal(frames):
    """Per-transition mean absolute RGB change, length F-1."""
    frames = np.asarray(frames, dtype=np.float32)
    if len(frames) < 2:
        return np.empty(0, dtype=np.float32)
    return np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2, 3))


def motion_sync(frames, meta):
    """Zero-lag correlation between visible motion and requested motion."""
    motion = visual_motion_signal(frames)
    control = np.asarray(meta["motion_control"], dtype=np.float32)[:-1]
    n = min(len(motion), len(control))
    motion, control = motion[:n], control[:n]
    if n < 3 or np.std(motion) < 1e-7 or np.std(control) < 1e-7:
        return float("nan")
    return float(pearsonr(motion, control).statistic)


def signal_sync(motion, control, method="pearson"):
    """Correlation for any output-side transition signal."""
    motion = np.asarray(motion, dtype=np.float32)
    control = np.asarray(control, dtype=np.float32)
    n = min(len(motion), len(control))
    motion, control = motion[:n], control[:n]
    if n < 3 or np.std(motion) < 1e-7 or np.std(control) < 1e-7:
        return float("nan")
    fn = pearsonr if method == "pearson" else spearmanr
    return float(fn(motion, control).statistic)


def perceptual_motion_signal(frames, batch_size=8):
    """Per-transition LPIPS distance, length F-1."""
    global _LPIPS_MODEL
    import lpips, torch
    if len(frames) < 2:
        return np.empty(0, dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device).eval()
    x = torch.from_numpy(np.asarray(frames, dtype=np.float32)).permute(0, 3, 1, 2)
    x = x.to(device) * 2.0 - 1.0
    values = []
    with torch.no_grad():
        for i in range(0, len(x) - 1, batch_size):
            stop = min(i + batch_size, len(x) - 1)
            a, b = x[i:stop], x[i + 1:stop + 1]
            values.extend(_LPIPS_MODEL(a, b).flatten().cpu().numpy().tolist())
    return np.asarray(values, dtype=np.float32)


def lagged_motion_sync(frames, meta, max_lag=8):
    """Return the lag with maximum visible-motion/control correlation.

    Positive lag means the visual response follows the audio request.
    """
    motion = visual_motion_signal(frames)
    control = np.asarray(meta["motion_control"], dtype=np.float32)[:-1]
    n = min(len(motion), len(control))
    motion, control = motion[:n], control[:n]
    scores = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x, y = motion[lag:], control[:-lag]
        elif lag < 0:
            x, y = motion[:lag], control[-lag:]
        else:
            x, y = motion, control
        if len(x) < 3 or np.std(x) < 1e-7 or np.std(y) < 1e-7:
            scores[lag] = float("nan")
        else:
            scores[lag] = float(pearsonr(x, y).statistic)
    valid = [(lag, value) for lag, value in scores.items() if np.isfinite(value)]
    if not valid:
        return {"best_lag_frames": float("nan"), "best_lag_corr": float("nan")}
    lag, value = max(valid, key=lambda item: item[1])
    return {"best_lag_frames": lag, "best_lag_corr": value}


def lpips_metric(frames, n_pairs=64, random_pairs=64, seed=0):
    """Mean LPIPS on (a) consecutive frames (flicker) and (b) random pairs (drift)."""
    import torch
    F = frames.shape[0]
    if F < 2:
        return {"lpips_flicker": float("nan"), "lpips_drift": float("nan")}
    rng = np.random.default_rng(seed)
    transition = perceptual_motion_signal(frames)
    flicker = float(transition.mean())
    p1 = rng.integers(0, F, size=min(random_pairs, F)); p2 = rng.integers(0, F, size=min(random_pairs, F))
    global _LPIPS_MODEL
    device = next(_LPIPS_MODEL.parameters()).device
    ia = torch.from_numpy(frames[p1].astype(np.float32)).permute(0, 3, 1, 2).to(device) * 2 - 1
    ib = torch.from_numpy(frames[p2].astype(np.float32)).permute(0, 3, 1, 2).to(device) * 2 - 1
    with torch.no_grad():
        drift = float(_LPIPS_MODEL(ia, ib).mean())
    return {"lpips_flicker": flicker, "lpips_drift": drift}


def summary(out_dir, fid_value=None):
    meta = load_meta(out_dir)
    frames = load_frames(out_dir)
    s = {"temporal_diff": temporal_diff(frames),
         "motion_sync": motion_sync(frames, meta)}
    s.update(lagged_motion_sync(frames, meta))
    if "control_condition" in meta:
        s["control_condition"] = str(meta["control_condition"])
    if fid_value is not None:
        s["fid"] = fid_value
    return s
