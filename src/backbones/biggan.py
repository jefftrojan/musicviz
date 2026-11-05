"""BigGAN backbone: pytorch-pretrained-biggan (op-for-op reimplementation of
DeepMind's BigGAN, Brock et al. 2018). CPU-friendly at 128x128.

Usage matches the library's quick-start: BigGAN.from_pretrained, truncated noise,
one-hot class vectors, optional truncation.
"""
import torch
import numpy as np
from pytorch_pretrained_biggan import BigGAN


class BigGANBackbone:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = BigGAN.from_pretrained(cfg["model"])  # cached in ~/.pytorch_pretrained_biggan/
        self.model.eval()
        self.truncation = float(cfg.get("truncation", 0.7))

    @torch.no_grad()
    def generate(self, z, class_labels):
        """z: (F,128) np array; class_labels: (F,1000) np array -> (F,3,128,128) uint8."""
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(dev)
        bs = int(self.cfg.get("batch", 8))
        res = self.model.config.output_dim
        out = np.empty((len(z), 3, res, res), dtype=np.float32)
        zv = torch.from_numpy(np.ascontiguousarray(z, dtype=np.float32)).to(dev)
        cv = torch.from_numpy(np.ascontiguousarray(class_labels, dtype=np.float32)).to(dev)
        for i in range(0, len(z), bs):
            img = self.model(zv[i:i + bs], cv[i:i + bs], self.truncation)
            out[i:i + bs] = ((img.cpu().numpy() + 1.0) / 2.0)
        return np.clip(out, 0, 1)
