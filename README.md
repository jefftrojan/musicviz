# Causal Audio Reactivity Across Pretrained Image Generators

Research code and artifacts for a controlled comparison of BigGAN, StyleGAN2,
and Stable Diffusion. The study asks whether the same requested audio-motion
trace causes visible, correctly timed change in each generator.

Semantic content is fixed within each sequence. The experiment manipulates only
the timing of a shared RMS/onset motion trace using five conditions:
`aligned`, `shifted`, `shuffled`, `flat`, and `matched_random`.

## Layout

```text
config.yaml             experiment settings and adapter calibration scales
run.py                  song × backbone × control-condition driver
src/audio_features.py   chroma, RMS, onset envelope, tempo
src/control.py          shared motion trace and causal interventions
src/mapping.py          BigGAN, StyleGAN2, and diffusion adapters
src/backbones/          generator inference implementations
metrics.py              visible motion, zero-lag sync, and lag analysis
colab/musicviz_baselines.ipynb self-contained GPU causal-matrix workflow
kaggle/musicviz_kaggle.ipynb Kaggle GPU causal-matrix workflow
analysis/full/          aggregate measurements, statistics, and figures
docs/                   public paired-comparison study and frozen stimuli
supabase/schema.sql     insert-only response database schema and policy
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader wordnet
```

BigGAN weights download on first use. StyleGAN2 uses NVIDIA's official
`stylegan2-ada-pytorch` repository and an FFHQ checkpoint configured in
`config.yaml`. Stable Diffusion uses the maintained SD 1.5 Hugging Face mirror.

The repository includes three CC BY 3.0 source tracks and a reproducible
`prepare_songs.py` step that creates the frozen study excerpts documented in
`songs/README.md`. The primary protocol uses 8-second excerpts at 4 fps
(32 frames), three seeds, and identical timestamps for every backbone and
control condition.

## Completed experiment

The full matrix contains 135 sequences: three songs, three seeds, three
backbones, and five causal timing conditions. Across the 27 paired experimental
units, aligned timing did not reliably outperform the controls in perceptual
motion synchrony (mean difference 0.0069, 95% bootstrap CI
[-0.0429, 0.0640], sign-flip p = 0.8273). Backbone-level estimates and all
per-run measurements are in `analysis/full/`.

## Paired-comparison study

The static study is published at
[jefftrojan.github.io/musicviz](https://jefftrojan.github.io/musicviz/). It
runs in the browser and submits anonymous responses to the configured research
database. If submission fails, it downloads a CSV backup instead.

### Online response storage

The study can submit completed sessions directly to Supabase and falls back to
a local CSV download if submission fails.

1. Create a Supabase project and run `supabase/schema.sql` in its SQL editor.
2. Copy the project URL and **publishable** key from the project's Connect/API
   settings into `docs/config.js`.
3. Never place a secret key or service-role key in the site or repository.
4. Download accumulated responses as CSV from the `study_responses` table in
   the Supabase dashboard.

The database grants anonymous visitors insert access only. Row-level security
denies reading, updating, and deleting responses from the public website.

## Run the causal matrix

Prepare and verify the frozen licensed excerpts:

```bash
python prepare_songs.py
```

Calibrate every requested backbone on the held-out signal. The command writes
candidate measurements, previews, and a guarded `config_calibrated.yaml`:

```bash
python calibrate.py --backbones biggan --frames 8 --target 0.02 \
  --output calibration/biggan_results.json
python calibrate.py --backbones stylegan2 --frames 8 --target 0.02 \
  --output calibration/stylegan2_results.json
python calibrate.py --backbones sd --frames 8 --target 0.02 \
  --output calibration/sd_results.json
```

StyleGAN2 and diffusion calibration should be run on a GPU. A `paper` run is
rejected when a requested backbone is absent from the calibrated-backbone list.

Run one aligned condition:

```bash
python run.py --backbones biggan --controls aligned --seeds 0 --variant base
```

Run all causal conditions:

```bash
python run.py --backbones biggan,stylegan2 --controls all --variant base
python run.py --backbones sd --controls all --variant base
```

The final resume-safe matrix and analysis commands are:

```bash
python run.py --config config_calibrated.yaml \
  --backbones biggan,stylegan2,sd --controls all --variant paper --video --resume
python analyze.py --config config_calibrated.yaml --variant paper \
  --backbones biggan,stylegan2,sd --content-model dinov2
```

`analyze.py` refuses an incomplete matrix by default and writes a missing-run
manifest, per-run LPIPS/pixel/DINO metrics, lag curves, paired contrasts,
bootstrap confidence intervals, sign-flip tests, and publication figures.

Each output directory contains RGB frames and `meta.npz` with the raw envelope,
conditioned motion request, seed, control label, timestamps, and adapter state.

## Evaluation contract

`motion_sync` correlates LPIPS perceptual change between consecutive generated
frames with the saved motion request. It does not use latent step magnitude.
The primary result is the paired difference between aligned and control
conditions for each song, seed, and backbone. Pixel-motion synchrony and DINO
content consistency are reported as complementary diagnostics.

Do not compare FID values across FFHQ, ImageNet, and Stable Diffusion outputs as
if they measured one common target distribution. Use blinded aesthetic ratings
and within-backbone quality or consistency measures instead.
