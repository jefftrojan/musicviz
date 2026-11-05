"""Main driver: for each song x backbone x mapping-variant, generate a frame
sequence + save metadata needed by the metrics.

Usage:
  python run.py --config config.yaml --backbones biggan,stylegan2 --variant base
  python run.py --config config.yaml --backbones sd --variant base --smoothing 0.0

Outputs (per run): output/<song>_<backbone>_<variant>_seed<N>_<condition>/
  frames/0000.png ...   (or a single mp4 via --video)
  meta.npz              (chroma, class_vectors or w-traj, strength, rms, beat, times)
"""
import argparse, os, sys, time, resource, subprocess
import numpy as np, yaml, imageio
import imageio_ffmpeg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from audio_features import extract
from control import build_control, CONTROL_CONDITIONS
from mapping import BigGANMapping, StyleGAN2Mapping, SDMapping


def load_backbone(name, cfg):
    if name == "biggan":
        from backbones.biggan import BigGANBackbone
        return BigGANBackbone(cfg["backbones"]["biggan"])
    if name == "stylegan2":
        from backbones.stylegan2 import StyleGAN2Backbone
        return StyleGAN2Backbone(cfg["backbones"]["stylegan2"])
    if name == "sd":
        from backbones.sd import SDBackbone
        return SDBackbone(cfg["backbones"]["sd"])
    raise ValueError(name)


def run_one(song, name, cfg, backbone_name, variant, control_condition="aligned",
            smoothing_override=None, seed_override=None, resume=False,
            backbone=None):
    acfg, mcfg = cfg["audio"], dict(cfg["mapping"])
    if smoothing_override is not None:
        mcfg["latent_smoothing"] = smoothing_override
    feats = extract(song, acfg["fps"], acfg["seconds"], acfg["start"], acfg["sr"], acfg["hop"])
    max_frames = cfg.get("max_frames")
    if max_frames is not None:
        max_frames = int(max_frames)
        feats = {key: (value[:max_frames] if isinstance(value, np.ndarray) else value)
                 for key, value in feats.items()}
    seed = int(mcfg.get("seed", 0) if seed_override is None else seed_override)
    mcfg["seed"] = seed
    ccfg = dict(cfg["control"])
    ccfg["fps"] = acfg["fps"]
    ccfg["seed"] = seed
    raw_motion, motion_control = build_control(feats, ccfg, control_condition)
    feats["motion_control"] = motion_control
    classes = mcfg["chroma_classes"]
    out_dir = f"output/{name}_{backbone_name}_{variant}_seed{seed}_{control_condition}"
    expected = len(feats["times"])
    if resume and os.path.exists(os.path.join(out_dir, "meta.npz")):
        existing = [item for item in os.listdir(os.path.join(out_dir, "frames"))
                    if item.endswith(".png")]
        if len(existing) == expected:
            print(f"[cached] {out_dir}")
            return out_dir
    os.makedirs(os.path.join(out_dir, "frames"), exist_ok=True)

    backbone_cfg = dict(cfg)
    backbone_cfg["backbones"] = {key: dict(value) for key, value in cfg["backbones"].items()}
    backbone_cfg["backbones"][backbone_name]["seed"] = seed
    bb = backbone if backbone is not None else load_backbone(backbone_name, backbone_cfg)
    if backbone_name == "sd":
        bb.seed = seed
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    meta = {"chroma": feats["chroma"], "rms": feats["rms"], "beat": feats["beat"],
            "times": feats["times"], "tempo": feats["tempo"],
            "raw_motion": raw_motion, "motion_control": motion_control,
            "control_condition": np.array(control_condition), "seed": np.array(seed),
            "config_yaml": np.array(yaml.safe_dump(cfg, sort_keys=False)),
            "model_id": np.array(str(cfg["backbones"][backbone_name].get(
                "model", cfg["backbones"][backbone_name].get("weights", ""))))}

    if backbone_name == "biggan":
        adapter_cfg = dict(mcfg)
        adapter_cfg["adapter_motion_scale"] = cfg["backbones"]["biggan"].get("motion_scale", 1.0)
        m = BigGANMapping(feats, adapter_cfg, classes, seed)
        t0 = time.time()
        imgs = bb.generate(m.z.traj, m.class_vectors)
        meta["z"] = m.z.traj
        meta["class_vectors"] = m.class_vectors
        meta["adapter_motion_scale"] = np.array(adapter_cfg["adapter_motion_scale"])
    elif backbone_name == "stylegan2":
        adapter_cfg = dict(mcfg)
        adapter_cfg["adapter_motion_scale"] = cfg["backbones"]["stylegan2"].get("motion_scale", 1.0)
        m = StyleGAN2Mapping(feats, adapter_cfg, w_dim=bb.latent_dim, seed=seed)
        t0 = time.time()
        imgs = bb.generate(m.w.traj)
        meta["w"] = m.w.traj
        meta["adapter_motion_scale"] = np.array(adapter_cfg["adapter_motion_scale"])
    elif backbone_name == "sd":
        sd_mcfg = dict(mcfg)
        sd_mcfg.update(cfg["backbones"]["sd"])
        sd_mcfg["adapter_motion_scale"] = cfg["backbones"]["sd"].get("motion_scale", 1.0)
        m = SDMapping(feats, sd_mcfg, classes)
        t0 = time.time()
        imgs = bb.generate(m.prompts, m.strength)
        meta["strength"] = m.strength
        meta["prompts"] = np.array(m.prompts)
        meta["adapter_motion_scale"] = np.array(sd_mcfg["adapter_motion_scale"])
        meta["strength_min"] = np.array(sd_mcfg["strength_min"])
        meta["strength_max"] = np.array(sd_mcfg["strength_max"])

    dt = time.time() - t0
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_peak = max(rss_before, rss_after)
    peak_memory_mb = (rss_peak / (1024.0 ** 2) if sys.platform == "darwin"
                      else rss_peak / 1024.0)
    try:
        import torch
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
    except Exception:
        pass
    meta["elapsed_seconds"] = np.array(dt)
    meta["s_per_frame"] = np.array(dt / len(imgs))
    meta["peak_memory_mb"] = np.array(peak_memory_mb)
    meta["backbone"] = np.array(backbone_name)
    meta["song"] = np.array(name)
    for i, im in enumerate(imgs):
        if im.ndim == 3 and im.shape[0] in (1, 3, 4):
            im = np.moveaxis(im, 0, -1)
        imageio.imwrite(os.path.join(out_dir, "frames", f"{i:04d}.png"),
                        (np.clip(im, 0, 1) * 255).astype(np.uint8))
    if cfg.get("video", False):
        video_frames = [np.moveaxis(im, 0, -1)
                        if im.ndim == 3 and im.shape[0] in (1, 3, 4) else im
                        for im in imgs]
        silent = os.path.join(out_dir, "visual_only.mp4")
        imageio.mimsave(silent,
                        [(np.clip(im, 0, 1) * 255).astype(np.uint8)
                         for im in video_frames],
                        fps=int(acfg["fps"]), codec="libx264")
        clip = os.path.join(out_dir, "clip.mp4")
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-ss", str(acfg["start"]),
                   "-t", str(acfg["seconds"]), "-i", song, "-i", silent,
                   "-map", "1:v:0", "-map", "0:a:0", "-c:v", "copy",
                   "-c:a", "aac", "-shortest", clip]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE)
        os.remove(silent)
    np.savez_compressed(os.path.join(out_dir, "meta.npz"), **meta)
    print(f"[{backbone_name}/{variant}/{control_condition}] {name}: {len(imgs)} frames in {dt:.0f}s "
          f"({dt/len(imgs):.2f}s/frame) -> {out_dir}")
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--backbones", default="biggan")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--controls", default="aligned",
                    help="comma list: aligned,shifted,shuffled,flat,matched_random, or all")
    ap.add_argument("--smoothing", type=float, default=None,
                    help="override mapping.latent_smoothing (anti-flicker ablation)")
    ap.add_argument("--songs", default=None, help="comma list; default from config")
    ap.add_argument("--seeds", default=None,
                    help="comma list; default from config.seeds or mapping.seed")
    ap.add_argument("--video", action="store_true", help="also write mp4s")
    ap.add_argument("--resume", action="store_true",
                    help="skip complete output directories")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.video:
        cfg["video"] = True
    songs = args.songs.split(",") if args.songs else cfg["songs"]
    controls = (list(CONTROL_CONDITIONS) if args.controls == "all"
                else [item.strip() for item in args.controls.split(",")])
    requested_backbones = [item.strip() for item in args.backbones.split(",")]
    if args.variant == "paper":
        completed = set(cfg.get("calibration", {}).get("completed_backbones", []))
        missing = [name for name in requested_backbones if name not in completed]
        if missing:
            raise ValueError("paper run requires completed calibration for: "
                             + ", ".join(missing))
    seeds = ([int(item.strip()) for item in args.seeds.split(",")]
             if args.seeds else [int(value) for value in
             cfg.get("seeds", [cfg["mapping"].get("seed", 0)])])
    for bb in requested_backbones:
        backbone = load_backbone(bb, cfg)
        for song in songs:
            for seed in seeds:
                for control_condition in controls:
                    name = os.path.splitext(os.path.basename(song))[0]
                    run_one(song, name, cfg, bb, args.variant,
                            control_condition, args.smoothing, seed, args.resume,
                            backbone)


if __name__ == "__main__":
    main()
