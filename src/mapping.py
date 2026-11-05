"""Generator adapters for a shared, experimentally manipulated motion trace."""
import numpy as np


class LatentWalk:
    """Random walk whose requested step norm is shared across GAN adapters."""

    def __init__(self, dim, feats, mcfg, seed=0):
        rng = np.random.default_rng(seed)
        n = len(feats["times"])
        control = np.asarray(feats["motion_control"], dtype=np.float32)
        base = float(mcfg["base_walk_speed"])
        gain = float(mcfg.get("motion_gain", 1.0))
        adapter_scale = float(mcfg.get("adapter_motion_scale", 1.0))
        speed = adapter_scale * base * (1.0 + gain * control)
        z = np.empty((n, dim), dtype=np.float32)
        z[0] = rng.normal(size=dim)
        for i in range(1, n):
            d = rng.normal(size=dim)
            d /= (np.linalg.norm(d) + 1e-9)
            z[i] = z[i - 1] + speed[i - 1] * d
        a = float(mcfg.get("latent_smoothing", 0.0))
        if not 0.0 <= a <= 1.0:
            raise ValueError("latent_smoothing must be in [0, 1]")
        if a > 0.0:
            s = z.copy()
            for i in range(1, n):
                s[i] = a * s[i - 1] + (1.0 - a) * z[i]
            z = s
        self.traj = z  # (n, dim)
        self.requested_step = speed[:-1]


class BigGANMapping:
    """BigGAN motion adapter with fixed semantics for the causal experiment."""

    def __init__(self, feats, mcfg, class_names, seed=0):
        from pytorch_pretrained_biggan import one_hot_from_names
        n = len(feats["times"])
        if mcfg.get("semantic_mode", "fixed") == "chroma":
            self.onehots = one_hot_from_names(list(class_names), batch_size=len(class_names))
            C = feats["chroma"].copy()
            C = C / (C.max(0) + 1e-9)
            sensitivity = max(float(mcfg.get("pitch_sensitivity", 1.0)), 0.05)
            C = C ** sensitivity
            C = C / (C.sum(1, keepdims=True) + 1e-9)
            self.class_vectors = C @ self.onehots
        else:
            fixed = mcfg.get("fixed_biggan_class", "lighthouse")
            self.class_vectors = one_hot_from_names(fixed, batch_size=n)
        self.z = LatentWalk(128, feats, mcfg, seed)


class StyleGAN2Mapping:
    """StyleGAN motion adapter; the official generator maps this z walk to w."""

    def __init__(self, feats, mcfg, w_dim=512, seed=0):
        self.w = LatentWalk(w_dim, feats, mcfg, seed)


class SDMapping:
    """Diffusion adapter with fixed semantics and shared motion control."""

    def __init__(self, feats, mcfg, class_names):
        n = len(feats["times"])
        prompt = mcfg.get("fixed_prompt", "an abstract cinematic landscape")
        prompt += mcfg.get("prompt_style", "")
        self.prompts = [prompt] * n
        control = np.asarray(feats["motion_control"], dtype=np.float32)
        control = np.clip(float(mcfg.get("adapter_motion_scale", 1.0)) * control,
                          0.0, 1.0)
        span = mcfg["strength_max"] - mcfg["strength_min"]
        # strength[t] controls transition t-1 -> t; GAN transition t -> t+1
        # uses control[t], so shift the same request forward by one frame.
        transition_control = np.r_[control[0], control[:-1]]
        self.strength = mcfg["strength_min"] + span * transition_control
        self.strength = np.clip(self.strength, mcfg["strength_min"], mcfg["strength_max"])
