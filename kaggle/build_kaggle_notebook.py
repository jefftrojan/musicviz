#!/usr/bin/env python3
"""Build a Kaggle-native notebook from the validated causal Colab notebook."""
import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / "colab/musicviz_baselines.ipynb").read_text())
nb = deepcopy(source)


def set_cell(index, text):
    nb["cells"][index]["source"] = text.splitlines(True)


set_cell(0, """# Causal Audio Reactivity — Kaggle GPU Runner

This Kaggle-native runner completes the GPU-bound portion of the causal matrix.
The calibrated 45-sequence BigGAN subset was completed locally; use this
notebook for StyleGAN2 and Stable Diffusion.

Recommended version sequence:

1. `RUN_BACKBONES=['stylegan2']`, all three songs and seeds.
2. Attach version 1 as an input, then run `RUN_BACKBONES=['sd']` with song `[0]`.
3. Save two more SD versions for song indices `[1]` and `[2]`, attaching prior
   outputs each time.

Enable **Internet** and **GPU T4 x2** in Settings. Use **Save Version → Save &
Run All** so `/kaggle/working` is persisted. The code uses one T4 deliberately;
the second remains spare rather than introducing multi-process nondeterminism.
""")

set_cell(1, """# 0. Kaggle GPU setup and dependencies
import socket
try:
    socket.gethostbyname('pypi.org')
except OSError as exc:
    raise RuntimeError('Kaggle Internet is disabled. Open Settings → Session options, turn Internet ON, save, and restart the session.') from exc

# Kaggle's Python 3.13 image already ships a mutually compatible current
# Diffusers/Transformers stack. Do not downgrade it: older tokenizers releases
# have no Python 3.13 wheel and would roll back this entire install transaction.
%pip install -q librosa soundfile scipy Pillow tqdm imageio imageio-ffmpeg lpips pandas
%pip install -q --no-deps pytorch-pretrained-biggan
%pip install -q nltk

import nltk, os, sys, torch
nltk.download('wordnet', quiet=True)
assert torch.cuda.is_available(), 'Settings → Accelerator → GPU T4 x2, then restart.'
print('torch',torch.__version__,'| visible GPUs:',torch.cuda.device_count(),
      '| active:',torch.cuda.get_device_name(0))

STYLEGAN_REPO='/kaggle/working/stylegan2-ada-pytorch'
!test -d "$STYLEGAN_REPO" || git clone --depth 1 -q https://github.com/NVlabs/stylegan2-ada-pytorch "$STYLEGAN_REPO"

# StyleGAN2-ADA's optional CUDA extensions target older PyTorch/Python builds.
# Kaggle's current Python 3.13 / PyTorch 2.11 image cannot compile them, and the
# upstream loader retries on every layer invocation. Select the repository's
# official PyTorch reference implementations before importing the model. This
# keeps the pretrained weights and generation equations unchanged; it only
# avoids the incompatible fused-kernel optimization.
from pathlib import Path
for rel in ('torch_utils/ops/bias_act.py', 'torch_utils/ops/upfirdn2d.py'):
    op_path=Path(STYLEGAN_REPO)/rel
    source=op_path.read_text()
    patched=source.replace("impl='cuda'", "impl='ref'")
    assert patched != source or "impl='ref'" in source, f'Could not patch {op_path}'
    op_path.write_text(patched)
print('StyleGAN2 ops: forced official reference kernels for Kaggle compatibility')

sys.path.insert(0,STYLEGAN_REPO)
STYLEGAN_URL='https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
print('Setup complete. Keep Internet enabled for model downloads.')
""")

set_cell(2, """## 1. Frozen licensed excerpts

Create a private Kaggle Dataset from `musicviz-study-audio.zip` and attach it
through **Add Input**. The code finds the three frozen WAV files by name. It
will not silently substitute synthetic audio in a paper run.
""")

set_cell(3, """from pathlib import Path
import json
import numpy as np, soundfile as sf

INPUT_ROOT=Path('/kaggle/input')
STUDY_NAMES=['calm_blue_lake.wav','acoustic_shuffle.wav','heavy_drums_bass.wav']

def unique_input(name):
    matches=sorted(INPUT_ROOT.rglob(name))
    if not matches: raise FileNotFoundError(f'Attach the study-audio dataset; missing {name}')
    if len(matches)>1: print('Multiple matches for',name,'using',matches[0])
    return str(matches[0])

SONGS=[unique_input(name) for name in STUDY_NAMES]
PRIOR_METRICS={p.parent.name:p for p in INPUT_ROOT.rglob('metrics.json')}
PRIOR_CALIBRATIONS=sorted(INPUT_ROOT.rglob('calibration.json'))
print('Songs:',SONGS)
print('Attached completed runs:',len(PRIOR_METRICS),'| calibration files:',len(PRIOR_CALIBRATIONS))
""")

set_cell(5, """# 2. Chunk configuration — edit only these three selectors between versions.
RUN_BACKBONES=['stylegan2']   # next stages: ['sd']
RUN_SONG_INDICES=[0,1,2]     # for SD use one of [0], [1], [2]
RUN_SEEDS=[0,1,2]

PAPER_RUN=True
CALIBRATED_SCALES={'biggan':2.0}  # measured locally: median LPIPS .01481
CALIBRATED_SD_INTERVAL=None
CALIBRATED_SD_BLEND_GAIN=None

# Inherit frozen GPU calibration from any attached earlier notebook output.
# Reject pilot calibrations that missed the target badly; otherwise a boundary
# result (such as StyleGAN2 scale 4.0) would be treated as final forever.
for calibration_path in PRIOR_CALIBRATIONS:
    data=json.loads(calibration_path.read_text())
    target=float(data.get('target_median_lpips',.02))
    results=data.get('new_results',{})
    for backbone,scale in data.get('motion_scales',{}).items():
        selected=results.get(backbone,{}).get('selected')
        median=None if selected is None else selected.get('median_lpips')
        if median is not None and not (.5*target <= float(median) <= 1.5*target):
            print('Ignoring under/over-calibrated',backbone,'from',calibration_path,
                  '| median LPIPS:',median,'target:',target)
            continue
        CALIBRATED_SCALES[backbone]=scale
    if data.get('sd_interval') is not None:
        CALIBRATED_SD_INTERVAL=tuple(data['sd_interval'])
    if data.get('sd_blend_gain') is not None:
        CALIBRATED_SD_BLEND_GAIN=float(data['sd_blend_gain'])

FPS,SECONDS,START=4,8,0
N_FRAMES=FPS*SECONDS
ALL_CONDITIONS=['aligned','shifted','shuffled','flat','matched_random']
SELECTED_SONGS=[SONGS[i] for i in RUN_SONG_INDICES]
SEEDS=list(RUN_SEEDS); BACKBONES=list(RUN_BACKBONES); CONDITIONS=ALL_CONDITIONS
OUTPUT_ROOT=Path('/kaggle/working/musicviz_causal'); OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)

FIXED_BIGGAN_CLASS='lighthouse'
FIXED_PROMPT='an abstract cinematic landscape, cinematic, soft lighting, high detail'
NEGATIVE_PROMPT='blurry, text, watermark, low quality'
BASE_WALK_SPEED,MOTION_GAIN=.08,2.0
MOTION_SCALE={'biggan':1.0,'stylegan2':1.0,'sd':1.0}; MOTION_SCALE.update(CALIBRATED_SCALES)
TRUNCATION,STYLEGAN_RESOLUTION=.7,256
SD_INIT_STEPS,SD_STEPS,SD_CFG,SD_STRENGTH=28,1000,6.5,(.001,.002)
SD_BLEND_GAIN=1.0
if CALIBRATED_SD_INTERVAL is not None: SD_STRENGTH=tuple(CALIBRATED_SD_INTERVAL)
if CALIBRATED_SD_BLEND_GAIN is not None: SD_BLEND_GAIN=float(CALIBRATED_SD_BLEND_GAIN)

planned=len(SELECTED_SONGS)*len(SEEDS)*len(CONDITIONS)*len(BACKBONES)
print('Chunk:',BACKBONES,RUN_SONG_INDICES,'| planned sequences:',planned)
""")

# Replace calibration cell. It inherits prior values and calibrates only missing
# settings for the selected backbone.
set_cell(13, """frozen_scales=dict(CALIBRATED_SCALES)
CALIBRATION_TARGET=.02
CALIBRATION_RESULTS={}
missing_gan=[b for b in BACKBONES if b in ('biggan','stylegan2') and b not in frozen_scales]
need_sd='sd' in BACKBONES and (CALIBRATED_SD_INTERVAL is None or CALIBRATED_SD_BLEND_GAIN is None)
RUN_CALIBRATION=bool(missing_gan or need_sd)

if RUN_CALIBRATION:
    cal_control=np.array([0,.15,.85,.30,1.0,.45,.70,.20],dtype='float32')
    for backbone in BACKBONES:
        candidates=[]
        if backbone=='sd' and need_sd:
            SD_STRENGTH=(.001,.002)
            for blend_gain in (.25,.40,.55,.70,.85,1.0):
                SD_BLEND_GAIN=blend_gain; frames,_=GENERATORS[backbone](cal_control,991)
                median=float(np.median(lpips_signal(frames)))
                candidates.append({'strength_min':SD_STRENGTH[0],'strength_max':SD_STRENGTH[1],
                                   'blend_gain':blend_gain,
                                   'median_lpips':median,'absolute_error':abs(median-CALIBRATION_TARGET)})
            selected=min(candidates,key=lambda row:row['absolute_error'])
            if not (.5*CALIBRATION_TARGET <= selected['median_lpips'] <= 1.5*CALIBRATION_TARGET):
                raise RuntimeError(
                    f'{backbone} calibration did not bracket the target: {selected}; '
                    f'candidates={candidates}. Adjust the interval grid before running the paper matrix.')
            SD_STRENGTH=(selected['strength_min'],selected['strength_max'])
            SD_BLEND_GAIN=float(selected['blend_gain'])
            CALIBRATED_SD_INTERVAL=SD_STRENGTH
            CALIBRATED_SD_BLEND_GAIN=SD_BLEND_GAIN
        elif backbone in missing_gan:
            # The reference-kernel StyleGAN2 path needs a larger latent scale
            # than BigGAN. Include enough range to bracket the LPIPS target.
            for scale in (.5,1.,2.,4.,8.,16.,32.):
                MOTION_SCALE[backbone]=scale; frames,_=GENERATORS[backbone](cal_control,991)
                median=float(np.median(lpips_signal(frames)))
                candidates.append({'motion_scale':scale,'median_lpips':median,
                                   'absolute_error':abs(median-CALIBRATION_TARGET)})
            selected=min(candidates,key=lambda row:row['absolute_error'])
            if not (.5*CALIBRATION_TARGET <= selected['median_lpips'] <= 1.5*CALIBRATION_TARGET):
                raise RuntimeError(
                    f'{backbone} calibration did not bracket the target: {selected}. '
                    'Expand the candidate range before running the paper matrix.')
            MOTION_SCALE[backbone]=selected['motion_scale']; frozen_scales[backbone]=selected['motion_scale']
        else:
            continue
        CALIBRATION_RESULTS[backbone]={'candidates':candidates,'selected':selected}
        print(backbone,'selected',selected)

CALIBRATED_SCALES=dict(frozen_scales)
assert all(b not in ('biggan','stylegan2') or b in CALIBRATED_SCALES for b in BACKBONES)
assert 'sd' not in BACKBONES or (CALIBRATED_SD_INTERVAL is not None and CALIBRATED_SD_BLEND_GAIN is not None)
calibration_payload={'target_median_lpips':CALIBRATION_TARGET,'seed':991,'frames':8,
                     'motion_scales':CALIBRATED_SCALES,
                     'sd_interval':list(CALIBRATED_SD_INTERVAL) if CALIBRATED_SD_INTERVAL else None,
                     'sd_blend_gain':CALIBRATED_SD_BLEND_GAIN,
                     'new_results':CALIBRATION_RESULTS}
(OUTPUT_ROOT/'calibration.json').write_text(json.dumps(calibration_payload,indent=2))
print('Frozen scales:',CALIBRATED_SCALES,'| SD interval:',CALIBRATED_SD_INTERVAL,
      '| SD blend gain:',CALIBRATED_SD_BLEND_GAIN)
""")

set_cell(14, """## 7. Execute this resume-safe chunk

Runs present in an attached prior notebook output are treated as immutable
cache entries. New runs are written under `/kaggle/working/musicviz_causal`.
Commit each chunk with **Save Version → Save & Run All** and enable output
persistence.
""")

set_cell(15, """for song_path in SELECTED_SONGS:
    song_id=Path(song_path).stem; features=audio_request(song_path); raw=features['raw_motion']
    for backbone in BACKBONES:
        for seed in SEEDS:
            for condition in CONDITIONS:
                run_name=f'{song_id}_{backbone}_seed{seed}_{condition}'
                run_dir=OUTPUT_ROOT/run_name
                # A backbone calibrated in this session must not reuse frames
                # generated with a rejected scale from an earlier version.
                force_rerun=backbone in CALIBRATION_RESULTS
                if not force_rerun and run_name in PRIOR_METRICS:
                    RESULTS.append(json.loads(PRIOR_METRICS[run_name].read_text())); print('attached cache',run_name); continue
                metric_file=run_dir/'metrics.json'
                if not force_rerun and metric_file.exists():
                    RESULTS.append(json.loads(metric_file.read_text())); print('session cache',run_name); continue
                control=make_control(raw,condition,seed); print('running',run_name,flush=True)
                torch.cuda.reset_peak_memory_stats(); t0=time.time()
                frames,adapter_meta=GENERATORS[backbone](control,seed); elapsed=time.time()-t0
                peak_mb=torch.cuda.max_memory_allocated()/(1024**2)
                meta={**features,**adapter_meta,'motion_control':control,
                      'control_condition':np.array(condition),'seed':np.array(seed),
                      'peak_memory_mb':np.array(peak_mb),'s_per_frame':np.array(elapsed/len(frames))}
                result=save_run(run_dir,frames,meta,elapsed,backbone,song_id,seed,condition,song_path)
                result['peak_memory_mb']=peak_mb; metric_file.write_text(json.dumps(result,indent=2,allow_nan=True))
                RESULTS.append(result); print('done',run_name,'s/frame',round(result['s_per_frame'],3),flush=True)
                del frames; torch.cuda.empty_cache()
print('Chunk complete:',len(RESULTS),'rows including attached cache entries')
""")

set_cell(16, """## 8. Export this version

Kaggle persists `/kaggle/working` when the notebook is committed. Attach this
notebook version as an input to the next chunk. Do not rely on downloading a
large ZIP through the browser.
""")

set_cell(17, """# Combine metrics visible from attached versions and this chunk.
all_metrics={p.parent.name:json.loads(p.read_text()) for p in INPUT_ROOT.rglob('metrics.json')}
for p in OUTPUT_ROOT.rglob('metrics.json'): all_metrics[p.parent.name]=json.loads(p.read_text())
df=pd.DataFrame(all_metrics.values())
if len(df):
    display(df.sort_values(['name','song','seed','control_condition']).round(4))
    df.to_csv(OUTPUT_ROOT/'metrics_visible.csv',index=False)
    controls=df[df.control_condition.isin(['shifted','shuffled','matched_random'])]
    aligned=df[df.control_condition.eq('aligned')][['name','song','seed','motion_sync']].rename(columns={'motion_sync':'aligned_sync'})
    control_mean=controls.groupby(['name','song','seed'],as_index=False).motion_sync.mean().rename(columns={'motion_sync':'control_sync'})
    paired=aligned.merge(control_mean,on=['name','song','seed'],how='left')
    paired['delta_sync']=paired.aligned_sync-paired.control_sync
    paired.to_csv(OUTPUT_ROOT/'paired_visible.csv',index=False); display(paired.round(4))
print('Saved output under',OUTPUT_ROOT)
""")

nb["metadata"].pop("colab", None)
nb["metadata"]["kaggle"] = {"accelerator": "gpu", "internet": True}
nb["metadata"]["accelerator"] = "GPU"
target = Path(__file__).with_name("musicviz_kaggle.ipynb")
target.write_text(json.dumps(nb, indent=1) + "\n")
print(f"wrote {target} ({len(nb['cells'])} cells)")
