"""Latent-diffusion backbone (Stable Diffusion 1.5, 512px) via diffusers.

This is the "modern" backbone of the 2021 article's pipeline. It is the most
expensive on CPU (~2-5 min/frame) -> run on Colab T4 (included in the free
Colab) or any GPU. ~30s/frame on a T4 is comfortable for 32 frames.

Protocol: an img2img CHAIN - frame t+1 is generated from frame t with
init_image=frame_t and strength=SDMapping.strength[t+1] (audio-driven).
This is the diffusion analogue of "volume/tempo drive the noise vector":
stronger audio -> bigger drift between frames.
"""
import torch
from PIL import Image
import numpy as np


class SDBackbone:
    def __init__(self, cfg):
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        self.cfg = cfg
        self.text_pipe = StableDiffusionPipeline.from_pretrained(
            cfg["model"], safety_checker=None)
        self.pipe = StableDiffusionImg2ImgPipeline(**self.text_pipe.components)
        if torch.cuda.is_available():
            self.text_pipe.to("cuda", torch.float16)
            self.pipe.to("cuda", torch.float16)
        self.negative = cfg.get("negative", "")
        self.steps = int(cfg.get("steps", 28))
        self.cfg_scale = float(cfg.get("cfg", 6.5))
        self.seed = int(cfg.get("seed", 0))

    @torch.no_grad()
    def generate(self, prompts, strengths):
        """prompts: list[str] len F; strengths: (F,) array; -> (F, 3, 512, 512) [0,1]."""
        F = len(prompts)
        out = np.empty((F, 3, 512, 512), dtype=np.float32)
        init = None
        for t in range(F):
            generator = torch.Generator(device="cpu").manual_seed(self.seed + t)
            kw = dict(prompt=prompts[t], negative_prompt=self.negative,
                      num_inference_steps=self.steps, guidance_scale=self.cfg_scale,
                      generator=generator)
            if init is not None and t > 0:
                kw["image"] = init
                kw["strength"] = float(strengths[t])
                img = self.pipe(**kw).images[0]
            else:
                img = self.text_pipe(**kw).images[0]
            arr = np.asarray(img, dtype=np.float32) / 255.0
            out[t] = np.moveaxis(arr, -1, 0)
            init = img
        return out
