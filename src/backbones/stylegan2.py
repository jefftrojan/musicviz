"""StyleGAN2-ADA inference using NVIDIA's official PyTorch repository."""
import os
import sys
import pickle
import torch
import numpy as np


class StyleGAN2Backbone:
    def __init__(self, cfg):
        self.cfg = cfg
        repo = os.path.abspath(cfg["repo"])
        if repo not in sys.path:
            sys.path.insert(0, repo)
        with open(cfg["weights"], "rb") as f:
            checkpoint = pickle.load(f)
        self.net = checkpoint["G_ema"].eval()
        self.truncation = float(cfg.get("truncation", 0.7))
        self.size = int(self.net.img_resolution)
        self.latent_dim = int(self.net.z_dim)
        self.output_resolution = int(cfg.get("output_resolution", self.size))

    def _synthesize(self, ws):
        """Run the official synthesis blocks, optionally stopping early."""
        x = img = None
        w_idx = 0
        for res in self.net.synthesis.block_resolutions:
            block = getattr(self.net.synthesis, f"b{res}")
            block_ws = ws.narrow(1, w_idx, block.num_conv + block.num_torgb)
            w_idx += block.num_conv
            x, img = block(x, img, block_ws, noise_mode="const")
            if res == self.output_resolution:
                return img
        return img

    @torch.no_grad()
    def generate(self, w):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.net.to(dev)
        bs = int(self.cfg.get("batch", 1))
        ws = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)).to(dev)
        labels = torch.zeros((len(w), self.net.c_dim), device=dev)
        s = self.output_resolution
        out = np.empty((len(w), 3, s, s), dtype=np.float32)
        for i in range(0, len(w), bs):
            mapped = self.net.mapping(ws[i:i + bs], labels[i:i + bs],
                                      truncation_psi=self.truncation)
            img = self._synthesize(mapped)
            out[i:i + bs] = ((img.cpu().numpy() + 1.0) / 2.0)
        return np.clip(out, 0, 1)
