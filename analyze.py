#!/usr/bin/env python3
"""Compute output-side metrics and paired causal statistics for completed runs."""
import argparse
import itertools
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from metrics import (load_frames, load_meta, perceptual_motion_signal,
                     signal_sync, visual_motion_signal)


NONFLAT_CONTROLS = ("shifted", "shuffled", "matched_random")
RUN_NAME_RE = re.compile(
    r"^(?P<song>.+)_(?P<backbone>biggan|stylegan2|sd)_(?P<variant>.+)_seed(?P<seed>\d+)_(?P<condition>aligned|shifted|shuffled|flat|matched_random)$")


class DinoEncoder:
    def __init__(self):
        from transformers import AutoImageProcessor, AutoModel
        model_id = "facebook/dinov2-small"
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode(self, frames, batch_size=16):
        embeddings = []
        for i in range(0, len(frames), batch_size):
            images = [Image.fromarray((frame * 255).astype(np.uint8))
                      for frame in frames[i:i + batch_size]]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                vectors = self.model(**inputs).last_hidden_state[:, 0]
                vectors = torch.nn.functional.normalize(vectors, dim=-1)
            embeddings.append(vectors.cpu().numpy())
        return np.concatenate(embeddings)


def lag_curve(motion, control, max_lag=8):
    rows = []
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            x, y = motion[lag:], control[:-lag]
        elif lag < 0:
            x, y = motion[:lag], control[-lag:]
        else:
            x, y = motion, control
        rows.append((lag, signal_sync(x, y)))
    return rows


def scalar(meta, key, default=np.nan):
    return float(meta[key]) if key in meta else default


def run_metrics(path, encoder, max_lag):
    frames, meta = load_frames(str(path)), load_meta(str(path))
    identity = RUN_NAME_RE.match(path.name)
    if identity is None:
        raise ValueError(f"cannot parse run identity from directory name: {path.name}")
    lpips_motion = perceptual_motion_signal(frames)
    pixel_motion = visual_motion_signal(frames)
    control = np.asarray(meta["motion_control"], dtype=np.float32)[:-1]
    curve = lag_curve(lpips_motion, control, max_lag)
    valid = [(lag, value) for lag, value in curve if np.isfinite(value)]
    best = max(valid, key=lambda item: item[1]) if valid else (np.nan, np.nan)
    row = {
        "path": str(path),
        "backbone": str(meta.get("backbone", "") or identity.group("backbone")),
        "song": str(meta.get("song", "") or identity.group("song")),
        "seed": int(meta.get("seed", identity.group("seed"))),
        "control_condition": str(meta.get("control_condition", "")),
        "motion_sync": signal_sync(lpips_motion, control),
        "motion_sync_spearman": signal_sync(lpips_motion, control, "spearman"),
        "pixel_motion_sync": signal_sync(pixel_motion, control),
        "lpips_motion_mean": float(lpips_motion.mean()),
        "lpips_motion_median": float(np.median(lpips_motion)),
        "lpips_motion_std": float(lpips_motion.std()),
        "temporal_diff": float(pixel_motion.mean()),
        "best_lag_frames": best[0], "best_lag_corr": best[1],
        "s_per_frame": scalar(meta, "s_per_frame"),
        "peak_memory_mb": scalar(meta, "peak_memory_mb"),
    }
    if encoder is not None:
        embeddings = encoder.encode(frames)
        row["dino_consecutive_consistency"] = float(
            np.sum(embeddings[:-1] * embeddings[1:], axis=1).mean())
        row["dino_first_frame_consistency"] = float(
            np.sum(embeddings[0] * embeddings[1:], axis=1).mean())
    lag_rows = [{"path": str(path), "backbone": row["backbone"],
                 "song": row["song"], "seed": row["seed"],
                 "control_condition": row["control_condition"],
                 "lag_frames": lag, "correlation": value}
                for lag, value in curve]
    return row, lag_rows


def bootstrap_ci(values, rng, draws=10000):
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values): return (np.nan, np.nan)
    means = np.mean(rng.choice(values, (draws, len(values)), replace=True), axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def sign_flip_p(values, rng, draws=10000):
    values = np.asarray(values, dtype=float); values = values[np.isfinite(values)]
    if not len(values): return np.nan
    observed = abs(values.mean())
    if len(values) <= 20:
        signs = np.asarray(list(itertools.product((-1, 1), repeat=len(values))))
        return float(np.mean(np.abs((signs * values).mean(axis=1)) >= observed))
    signs = rng.choice((-1, 1), (draws, len(values)))
    return float((np.sum(np.abs((signs * values).mean(axis=1)) >= observed) + 1) /
                 (draws + 1))


def paired_table(runs):
    aligned = runs[runs.control_condition == "aligned"]
    metric_columns = ("motion_sync", "motion_sync_spearman", "pixel_motion_sync",
                      "lpips_motion_mean", "temporal_diff")
    # The experimental unit is one backbone/song/seed. Average the three
    # non-flat nulls within that unit before forming a paired contrast; treating
    # them as independent pairs would pseudoreplicate the aligned observation.
    controls = (runs[runs.control_condition.isin(NONFLAT_CONTROLS)]
                .groupby(["backbone", "song", "seed"], as_index=False)
                [list(metric_columns)].mean())
    pairs = controls.merge(aligned, on=["backbone", "song", "seed"],
                           suffixes=("_control", "_aligned"))
    for metric in metric_columns:
        pairs[f"delta_{metric}"] = pairs[f"{metric}_aligned"] - pairs[f"{metric}_control"]
    return pairs


def write_summary(runs, pairs, output, rng):
    rows = []
    for name, group in [("overall", pairs), *pairs.groupby("backbone")]:
        values = group["delta_motion_sync"].to_numpy()
        low, high = bootstrap_ci(values, rng)
        rows.append({"backbone": name, "n_pairs": int(np.isfinite(values).sum()),
                     "mean_delta_motion_sync": float(np.nanmean(values)),
                     "ci_low": low, "ci_high": high,
                     "sign_flip_p": sign_flip_p(values, rng),
                     "median_s_per_frame": float(np.nanmedian(
                         runs.loc[runs.backbone == name, "s_per_frame"]))
                         if name != "overall" else float(np.nanmedian(runs.s_per_frame))})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "causal_summary.csv", index=False)
    lines = ["# Causal matrix analysis", "",
             f"Completed runs: {len(runs)}; non-flat aligned/control pairs: {len(pairs)}.", "",
             "| Backbone | Pairs | Mean Δ sync | 95% bootstrap CI | Sign-flip p |",
             "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['backbone']} | {row['n_pairs']} | "
                     f"{row['mean_delta_motion_sync']:.3f} | "
                     f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] | "
                     f"{row['sign_flip_p']:.4f} |")
    (output / "analysis_report.md").write_text("\n".join(lines) + "\n")
    return summary


def figures(runs, pairs, lags, output):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    labels, values = [], []
    for backbone, group in pairs.groupby("backbone"):
        labels.append(backbone); values.append(group.delta_motion_sync.dropna())
    if values:
        ax.boxplot(values, tick_labels=labels); ax.axhline(0, color="black", lw=.8)
        ax.set_ylabel("Aligned − control LPIPS-motion correlation")
        fig.tight_layout(); fig.savefig(output / "paired_sync_contrasts.pdf")
        fig.savefig(output / "paired_sync_contrasts.png", dpi=180)
    plt.close(fig)

    mean_lags = lags.groupby(["backbone", "control_condition", "lag_frames"],
                             as_index=False).correlation.mean()
    fig, axes = plt.subplots(1, max(1, runs.backbone.nunique()), figsize=(10, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (backbone, group) in zip(axes, mean_lags.groupby("backbone")):
        for condition, line in group.groupby("control_condition"):
            ax.plot(line.lag_frames, line.correlation, label=condition)
        ax.axvline(0, color="black", lw=.7); ax.set_title(backbone); ax.set_xlabel("Lag (frames)")
    axes[0].set_ylabel("LPIPS-motion correlation"); axes[-1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output / "lag_curves.pdf")
    fig.savefig(output / "lag_curves.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output")
    parser.add_argument("--variant", default="paper")
    parser.add_argument("--output", default="analysis")
    parser.add_argument("--config", default="config_calibrated.yaml")
    parser.add_argument("--backbones", default="biggan,stylegan2,sd")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--content-model", choices=("dinov2", "none"), default="dinov2")
    parser.add_argument("--reuse-frame-metrics", action="store_true",
                        help="reuse output/run_metrics.csv and add only content embeddings")
    parser.add_argument("--max-lag", type=int, default=8)
    args = parser.parse_args()
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    paths = sorted(path.parent for path in Path(args.root).glob(
        f"*_*_{args.variant}_seed*_*" + "/meta.npz"))
    if not paths:
        raise SystemExit(f"no completed {args.variant!r} runs under {args.root}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    expected = set()
    for song in cfg["songs"]:
        song_id = Path(song).stem
        for backbone in [item.strip() for item in args.backbones.split(",")]:
            for seed in cfg["seeds"]:
                for condition in ("aligned", "shifted", "shuffled", "flat", "matched_random"):
                    expected.add(f"{song_id}_{backbone}_{args.variant}_seed{seed}_{condition}")
    present = {path.name for path in paths}
    missing = sorted(expected - present)
    pd.DataFrame({"missing_run": missing}).to_csv(output / "missing_runs.csv", index=False)
    if missing and not args.allow_incomplete:
        raise SystemExit(f"matrix incomplete: {len(missing)} of {len(expected)} runs missing; "
                         f"see {output / 'missing_runs.csv'}")
    paths = [path for path in paths if path.name in expected]
    if args.reuse_frame_metrics:
        metrics_path, lags_path = output / "run_metrics.csv", output / "lag_curves.csv"
        if args.content_model != "dinov2" or not metrics_path.exists() or not lags_path.exists():
            raise SystemExit("--reuse-frame-metrics requires existing metrics/lags and --content-model dinov2")
        runs, lags = pd.read_csv(metrics_path), pd.read_csv(lags_path)
        expected_paths = {str(path) for path in paths}
        if set(runs.path) != expected_paths:
            raise SystemExit("cached run_metrics.csv does not match the current complete matrix")
        for column in ("dino_consecutive_consistency", "dino_first_frame_consistency"):
            if column not in runs: runs[column] = np.nan
        encoder = DinoEncoder()
        for index, row in runs.iterrows():
            if np.isfinite(row["dino_consecutive_consistency"]):
                continue
            print(f"[DINO {index + 1}/{len(runs)}] {Path(row.path).name}", flush=True)
            embeddings = encoder.encode(load_frames(row.path))
            runs.loc[index, "dino_consecutive_consistency"] = float(
                np.sum(embeddings[:-1] * embeddings[1:], axis=1).mean())
            runs.loc[index, "dino_first_frame_consistency"] = float(
                np.sum(embeddings[0] * embeddings[1:], axis=1).mean())
            runs.to_csv(metrics_path, index=False)
        pairs = paired_table(runs); pairs.to_csv(output / "paired_contrasts.csv", index=False)
        summary = write_summary(runs, pairs, output, np.random.default_rng(20261104))
        figures(runs, pairs, lags, output)
        print(summary.to_string(index=False))
        return
    encoder = DinoEncoder() if args.content_model == "dinov2" else None
    rows, lag_rows = [], []
    for index, path in enumerate(paths, 1):
        print(f"[{index}/{len(paths)}] {path.name}", flush=True)
        row, curve = run_metrics(path, encoder, args.max_lag)
        rows.append(row); lag_rows.extend(curve)
    runs, lags = pd.DataFrame(rows), pd.DataFrame(lag_rows)
    runs.to_csv(output / "run_metrics.csv", index=False)
    lags.to_csv(output / "lag_curves.csv", index=False)
    pairs = paired_table(runs); pairs.to_csv(output / "paired_contrasts.csv", index=False)
    summary = write_summary(runs, pairs, output, np.random.default_rng(20261104))
    figures(runs, pairs, lags, output)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
