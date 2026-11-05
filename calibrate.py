#!/usr/bin/env python3
"""Calibrate adapter motion on a held-out synthetic trace using median LPIPS."""
import argparse
import copy
import json
from pathlib import Path

import imageio
import numpy as np
import yaml

from metrics import perceptual_motion_signal
from run import load_backbone
from src.audio_features import extract
from src.control import build_control
from src.mapping import BigGANMapping, SDMapping, StyleGAN2Mapping


GAN_SCALES = (0.5, 1.0, 2.0, 4.0)
SD_INTERVALS = ((0.05, 0.20), (0.10, 0.30), (0.20, 0.50), (0.30, 0.65))


def nhwc(images):
    images = np.asarray(images, dtype=np.float32)
    if images.ndim == 4 and images.shape[1] in (1, 3, 4):
        images = np.moveaxis(images, 1, -1)
    return np.clip(images, 0.0, 1.0)


def candidate_specs(backbone):
    if backbone == "sd":
        return [{"strength_min": low, "strength_max": high,
                 "label": f"strength_{low:.2f}_{high:.2f}"}
                for low, high in SD_INTERVALS]
    return [{"motion_scale": scale, "label": f"scale_{scale:g}"}
            for scale in GAN_SCALES]


def generate(backbone_name, backbone, feats, cfg, spec, seed):
    mapping_cfg = dict(cfg["mapping"])
    classes = mapping_cfg["chroma_classes"]
    if backbone_name == "biggan":
        mapping_cfg["adapter_motion_scale"] = spec["motion_scale"]
        mapping = BigGANMapping(feats, mapping_cfg, classes, seed)
        return backbone.generate(mapping.z.traj, mapping.class_vectors)
    if backbone_name == "stylegan2":
        mapping_cfg["adapter_motion_scale"] = spec["motion_scale"]
        mapping = StyleGAN2Mapping(feats, mapping_cfg, backbone.latent_dim, seed)
        return backbone.generate(mapping.w.traj)
    mapping_cfg.update(cfg["backbones"]["sd"])
    mapping_cfg.update(spec)
    mapping_cfg["adapter_motion_scale"] = 1.0
    mapping = SDMapping(feats, mapping_cfg, classes)
    backbone.seed = seed
    return backbone.generate(mapping.prompts, mapping.strength)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--backbones", default="biggan,stylegan2,sd")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--target", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=991)
    parser.add_argument("--output", default="calibration/results.json")
    parser.add_argument("--config-output", default="config_calibrated.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    cfg["audio"].update({"fps": 2, "seconds": args.frames / 2, "start": 0})
    signal = "songs/calibration.wav"
    feats = extract(signal, cfg["audio"]["fps"], cfg["audio"]["seconds"], 0,
                    cfg["audio"]["sr"], cfg["audio"]["hop"])
    control_cfg = dict(cfg["control"]); control_cfg["fps"] = cfg["audio"]["fps"]
    _, control = build_control(feats, control_cfg, "aligned")
    feats["motion_control"] = control

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    results = {"target_median_lpips": args.target, "frames": args.frames,
               "seed": args.seed, "signal": signal, "candidates": {}, "selected": {}}
    preview_root = output.parent / "previews"; preview_root.mkdir(exist_ok=True)

    for name in [item.strip() for item in args.backbones.split(",")]:
        model_cfg = copy.deepcopy(cfg)
        model_cfg["backbones"][name]["batch"] = 4
        backbone = load_backbone(name, model_cfg)
        rows = []
        for spec in candidate_specs(name):
            print(f"calibrating {name}: {spec['label']}", flush=True)
            frames = nhwc(generate(name, backbone, feats, model_cfg, spec, args.seed))
            signal_values = perceptual_motion_signal(frames)
            median = float(np.median(signal_values))
            row = {**spec, "median_lpips": median,
                   "mean_lpips": float(np.mean(signal_values)),
                   "absolute_error": abs(median - args.target)}
            rows.append(row)
            imageio.imwrite(preview_root / f"{name}_{spec['label']}_last.png",
                            (frames[-1] * 255).astype(np.uint8))
            print(f"  median LPIPS={median:.5f}", flush=True)
        selected = min(rows, key=lambda row: row["absolute_error"])
        results["candidates"][name] = rows
        results["selected"][name] = selected
        output.write_text(json.dumps(results, indent=2))
        del backbone

    config_output = Path(args.config_output)
    calibrated = yaml.safe_load(config_output.read_text() if config_output.exists()
                                else Path(args.config).read_text())
    for name, selected in results["selected"].items():
        if name == "sd":
            calibrated["backbones"][name]["strength_min"] = selected["strength_min"]
            calibrated["backbones"][name]["strength_max"] = selected["strength_max"]
            calibrated["backbones"][name]["motion_scale"] = 1.0
        else:
            calibrated["backbones"][name]["motion_scale"] = selected["motion_scale"]
    calibration = calibrated.setdefault("calibration", {})
    completed = set(calibration.get("completed_backbones", []))
    completed.update(results["selected"])
    calibration.update({"target_median_lpips": args.target,
                        "frames": args.frames, "seed": args.seed})
    calibration["completed_backbones"] = sorted(completed)
    result_files = calibration.setdefault("result_files", {})
    for name in results["selected"]:
        result_files[name] = str(output)
    config_output.write_text(yaml.safe_dump(calibrated, sort_keys=False))
    print("wrote", config_output, "and", output)


if __name__ == "__main__":
    main()
