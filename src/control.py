"""Shared causal audio-control traces used by every generator adapter."""
import numpy as np


CONTROL_CONDITIONS = ("aligned", "shifted", "shuffled", "flat", "matched_random")


def motion_envelope(feats, cfg):
    """Return the normalized shared motion request before control manipulation."""
    rms_weight = float(cfg.get("rms_weight", 0.7))
    onset_weight = float(cfg.get("onset_weight", 0.3))
    total = rms_weight + onset_weight
    if total <= 0:
        raise ValueError("control rms_weight + onset_weight must be positive")
    signal = (rms_weight * feats["rms"] + onset_weight * feats["beat"]) / total
    return np.clip(signal, 0.0, 1.0).astype(np.float32)


def conditioned_motion(signal, condition, fps, cfg, seed=0):
    """Create a causal-control condition while preserving length and scale."""
    signal = np.asarray(signal, dtype=np.float32)
    if condition not in CONTROL_CONDITIONS:
        raise ValueError(f"unknown control condition: {condition}")
    if condition == "aligned":
        return signal.copy()
    if condition == "shifted":
        shift = max(1, int(round(float(cfg.get("shift_seconds", 2.0)) * fps)))
        return np.roll(signal, shift)
    if condition == "flat":
        return np.full_like(signal, signal.mean())

    rng = np.random.default_rng(seed)
    if condition == "matched_random":
        return rng.permutation(signal)

    block = max(1, int(round(float(cfg.get("shuffle_block_seconds", 0.5)) * fps)))
    blocks = [signal[i:i + block] for i in range(0, len(signal), block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[i] for i in order]).astype(np.float32)


def build_control(feats, cfg, condition):
    raw = motion_envelope(feats, cfg)
    seed = int(cfg.get("seed", 0))
    conditioned = conditioned_motion(raw, condition, cfg["fps"], cfg, seed)
    return raw, conditioned
