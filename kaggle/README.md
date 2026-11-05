# Kaggle GPU execution

Use `musicviz_kaggle.ipynb` to reproduce the StyleGAN2 and Stable Diffusion
matrix. Kaggle settings must enable **Internet** and **GPU T4 x2**.

## One-time audio dataset

Upload `musicviz-study-audio.zip` as a private Kaggle Dataset and attach it to
the notebook with **Add Input**. The notebook searches `/kaggle/input` for the
three frozen WAV filenames and fails if any are absent.

## Version sequence

1. StyleGAN2: `RUN_BACKBONES=['stylegan2']`, song indices `[0,1,2]`.
2. Stable Diffusion song 1: attach the StyleGAN version, set backbone `['sd']`
   and indices `[0]`.
3. Repeat SD with `[1]` and `[2]`, attaching preceding outputs.

Use **Save Version → Save & Run All** for every chunk. Kaggle persists only
`/kaggle/working`; every version writes calibration, metrics, frames, MP4s, and
metadata there. Attached prior outputs are automatically treated as completed
runs and their calibration is inherited.

After all four versions complete, download their outputs into this repository's
`output/` directory and run:

```bash
python analyze.py --config config_calibrated.yaml --variant paper \
  --backbones biggan,stylegan2,sd --content-model dinov2
```
