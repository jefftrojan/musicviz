#!/usr/bin/env python3
"""Regenerate the Colab notebook for the causal audio-reactivity protocol."""
import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": source.splitlines(True)}


cells = [
md("""# Causal Audio Reactivity Across Pretrained Image Generators

This notebook is the Colab implementation of the redesigned experiment. It asks
whether a shared, correctly timed audio-motion request produces visible change
in BigGAN, StyleGAN2, and Stable Diffusion.

The primary comparison uses fixed semantic content, paired seeds, and five
conditions: **aligned**, **shifted**, **block-shuffled**, **flat**, and
**distribution-matched random**. Metrics are computed from rendered pixels and
LPIPS transitions—not from latent step sizes.

The default configuration runs a five-condition BigGAN smoke test. Set
`PAPER_RUN = True` only after adding three licensed songs and connecting
persistent Drive storage; the complete matrix is 135 sequences and Stable
Diffusion is expensive.
"""),
code("""# 0. GPU setup and dependencies
%pip install -q librosa soundfile numpy scipy Pillow tqdm imageio imageio-ffmpeg lpips pandas diffusers==0.29.2 transformers accelerate huggingface-hub
%pip install -q --no-deps pytorch-pretrained-biggan
%pip install -q nltk

import nltk, os, sys, torch
nltk.download('wordnet', quiet=True)
assert torch.cuda.is_available(), 'Choose Runtime → Change runtime type → T4 GPU.'
print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0))

# Official NVIDIA StyleGAN2-ADA implementation and checkpoint.
!test -d /content/stylegan2-ada-pytorch || git clone --depth 1 -q https://github.com/NVlabs/stylegan2-ada-pytorch /content/stylegan2-ada-pytorch
sys.path.insert(0, '/content/stylegan2-ada-pytorch')
STYLEGAN_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
print('Setup complete. Model weights download lazily when their backbone is selected.')
"""),
md("""## 1. Songs

Upload three licensed audio files for the paper run. If nothing is uploaded,
the notebook creates three synthetic signals for pipeline validation only; they
are not research stimuli.
"""),
code("""from google.colab import files
import numpy as np, soundfile as sf
from pathlib import Path

uploaded = files.upload()
SONGS = list(uploaded)

def synth_song(path, bpm, mode, seconds=12, sr=22050):
    t = np.arange(int(seconds * sr)) / sr
    y = 0.10 * np.sin(2*np.pi*(110 + 22*mode)*t)
    beat = 60 / bpm
    for k in range(int(seconds / beat)):
        start = int(k * beat * sr); n = min(int(.20*sr), len(y)-start)
        u = np.arange(n) / sr
        y[start:start+n] += (.65 if k % 4 == 0 else .35) * np.sin(2*np.pi*(55+8*mode)*u) * np.exp(-22*u)
    if mode == 1: y *= np.linspace(.35, 1.0, len(y))
    if mode == 2: y *= .65 + .35*np.sin(2*np.pi*.18*t)**2
    y = .85 * y / (np.max(np.abs(y)) + 1e-9)
    sf.write(path, y.astype(np.float32), sr)

if not SONGS:
    SONGS=[]
    for i,(bpm,mode) in enumerate([(82,0),(118,1),(154,2)],1):
        path=f'/content/synthetic_song{i}.wav'; synth_song(path,bpm,mode); SONGS.append(path)
    print('Using synthetic validation signals:', SONGS)
else:
    print('Uploaded songs:', SONGS)
"""),
md("""## 2. Experiment configuration

The paper contract is 3 songs × 3 seeds × 5 conditions × 3 backbones, with
32 frames at 4 fps. The smoke mode keeps the same control logic but limits the
selected songs, seeds, and backbones.
"""),
code("""PAPER_RUN = False
# BigGAN=2.0 was measured locally (median LPIPS .01481 for target .02).
CALIBRATED_SCALES = {'biggan':2.0}  # add StyleGAN2 after its GPU sweep
CALIBRATED_SD_INTERVAL = None  # paper run: e.g. (.10,.30)

FPS, SECONDS, START = 4, 8, 0
N_FRAMES = FPS * SECONDS
ALL_CONDITIONS = ['aligned','shifted','shuffled','flat','matched_random']

if PAPER_RUN:
    assert len(SONGS) >= 3, 'Upload three licensed songs before a paper run.'
    SELECTED_SONGS, SEEDS = SONGS[:3], [0,1,2]
    BACKBONES = ['biggan','stylegan2','sd']
else:
    SELECTED_SONGS, SEEDS = SONGS[:1], [0]
    BACKBONES = ['biggan']

CONDITIONS = ALL_CONDITIONS
OUTPUT_ROOT = Path('/content/musicviz_causal')  # use a mounted Drive path for long runs
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

FIXED_BIGGAN_CLASS = 'lighthouse'
FIXED_PROMPT = 'an abstract cinematic landscape, cinematic, soft lighting, high detail'
NEGATIVE_PROMPT = 'blurry, text, watermark, low quality'
BASE_WALK_SPEED, MOTION_GAIN = .08, 2.0
MOTION_SCALE = {'biggan':1.0, 'stylegan2':1.0, 'sd':1.0}  # replace after held-out calibration
if CALIBRATED_SCALES is not None: MOTION_SCALE.update(CALIBRATED_SCALES)
TRUNCATION, STYLEGAN_RESOLUTION = .7, 256
SD_STEPS, SD_CFG, SD_STRENGTH = 28, 6.5, (.30,.65)
if CALIBRATED_SD_INTERVAL is not None: SD_STRENGTH = tuple(CALIBRATED_SD_INTERVAL)

print('Planned sequences:', len(SELECTED_SONGS)*len(SEEDS)*len(CONDITIONS)*len(BACKBONES))
print('Frames/sequence:', N_FRAMES, '| backbones:', BACKBONES)
"""),
md("""## 3. Shared audio request and causal interventions

Every adapter receives the same normalized request
`u(t) = 0.7·RMS(t) + 0.3·onset(t)`. Controls alter timing while preserving the
paired seed and semantic condition.
"""),
code("""import librosa, numpy as np
from scipy.stats import pearsonr, spearmanr

def robust_norm(x):
    lo,hi=np.percentile(x,[5,99]); return np.clip((x-lo)/(hi-lo+1e-9),0,1)

def resample_feature(x, frame_times, query_times):
    x=np.asarray(x)
    if x.ndim == 1: return np.interp(query_times, frame_times, x)
    return np.stack([np.interp(query_times, frame_times, row) for row in x], axis=1)

def audio_request(path):
    y,sr=librosa.load(path,sr=22050,mono=True,offset=START,duration=SECONDS)
    hop=512; query=np.arange(N_FRAMES)/FPS
    rms=librosa.feature.rms(y=y,hop_length=hop)[0]
    onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
    tr=librosa.frames_to_time(np.arange(len(rms)),sr=sr,hop_length=hop)
    to=librosa.frames_to_time(np.arange(len(onset)),sr=sr,hop_length=hop)
    rms=robust_norm(np.interp(query,tr,rms)); onset=robust_norm(np.interp(query,to,onset))
    return {'times':query.astype('float32'),'rms':rms.astype('float32'),
            'onset':onset.astype('float32'),'raw_motion':(.7*rms+.3*onset).astype('float32')}

def make_control(raw, condition, seed):
    raw=np.asarray(raw,dtype='float32')
    if condition=='aligned': return raw.copy()
    if condition=='shifted': return np.roll(raw,max(1,round(2.0*FPS)))
    if condition=='flat': return np.full_like(raw,raw.mean())
    rng=np.random.default_rng(seed)
    if condition=='matched_random': return rng.permutation(raw)
    block=max(1,round(.5*FPS)); blocks=[raw[i:i+block] for i in range(0,len(raw),block)]
    return np.concatenate([blocks[i] for i in rng.permutation(len(blocks))]).astype('float32')

def latent_walk(dim, control, seed, scale):
    rng=np.random.default_rng(seed); speed=scale*BASE_WALK_SPEED*(1+MOTION_GAIN*control)
    z=np.empty((len(control),dim),dtype='float32'); z[0]=rng.normal(size=dim)
    for i in range(1,len(z)):
        d=rng.normal(size=dim); d/=np.linalg.norm(d)+1e-9; z[i]=z[i-1]+speed[i-1]*d
    return z

for c in ALL_CONDITIONS:
    test=make_control(np.linspace(0,1,80),c,0)
    assert test.shape==(80,) and np.isfinite(test).all()
print('Audio/control contract ready.')
"""),
md("""## 4. Output-side metrics and resumable storage

`motion_sync` correlates requested motion with rendered frame-to-frame change.
LPIPS transition distance is the preferred perceptual motion signal; pixel
difference is retained as a transparent diagnostic. Flat controls correctly
produce undefined correlation because the request has no variance.
"""),
code("""import json, time, imageio, imageio_ffmpeg, subprocess, pandas as pd, torch

_lpips_model=None
def lpips_signal(frames):
    global _lpips_model
    import lpips
    if _lpips_model is None: _lpips_model=lpips.LPIPS(net='alex').cuda().eval()
    x=torch.from_numpy(frames.astype('float32')).permute(0,3,1,2).cuda()*2-1
    vals=[]
    with torch.no_grad():
        for i in range(0,len(x)-1,8):
            stop=min(i+8,len(x)-1); vals.extend(_lpips_model(x[i:stop],x[i+1:stop+1]).flatten().cpu().tolist())
    return np.asarray(vals,dtype='float32')

def pixel_signal(frames): return np.abs(frames[1:].astype('float32')-frames[:-1]).mean((1,2,3))

def corr(a,b,kind='pearson'):
    n=min(len(a),len(b)); a,b=np.asarray(a[:n]),np.asarray(b[:n])
    if n<3 or a.std()<1e-8 or b.std()<1e-8: return float('nan')
    return float((pearsonr if kind=='pearson' else spearmanr)(a,b).statistic)

def lag_result(motion,control,max_lag=8):
    scores={}
    for lag in range(-max_lag,max_lag+1):
        a,b=(motion[lag:],control[:-lag]) if lag>0 else ((motion[:lag],control[-lag:]) if lag<0 else (motion,control))
        scores[lag]=corr(a,b)
    valid=[(k,v) for k,v in scores.items() if np.isfinite(v)]
    return max(valid,key=lambda kv:kv[1]) if valid else (None,float('nan'))

def save_run(run_dir,frames,meta,elapsed,backbone,song_id,seed,condition,song_path):
    run_dir.mkdir(parents=True,exist_ok=True); (run_dir/'frames').mkdir(exist_ok=True)
    for i,im in enumerate(frames): imageio.imwrite(run_dir/'frames'/f'{i:04d}.png',(np.clip(im,0,1)*255).astype('uint8'))
    silent=run_dir/'visual_only.mp4'; clip=run_dir/'clip.mp4'
    imageio.mimsave(silent,[(np.clip(im,0,1)*255).astype('uint8') for im in frames],fps=FPS,codec='libx264')
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-ss',str(START),'-t',str(SECONDS),'-i',str(song_path),
         '-i',str(silent),'-map','1:v:0','-map','0:a:0','-c:v','copy','-c:a','aac','-shortest',str(clip)]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE); silent.unlink()
    np.savez_compressed(run_dir/'meta.npz',**meta)
    pixel=pixel_signal(frames); perceptual=lpips_signal(frames); control=meta['motion_control'][:-1]
    lag,lag_corr=lag_result(perceptual,control)
    result={'name':backbone,'song':song_id,'seed':seed,'control_condition':condition,
            'motion_sync':corr(perceptual,control),'motion_sync_spearman':corr(perceptual,control,'spearman'),
            'pixel_motion_sync':corr(pixel,control),'lpips_motion_mean':float(perceptual.mean()),
            'temporal_diff':float(pixel.mean()),'best_lag_frames':lag,'best_lag_corr':lag_corr,
            's_per_frame':elapsed/len(frames)}
    (run_dir/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=True)); return result

RESULTS=[]
print('Metrics/storage ready:', OUTPUT_ROOT)
"""),
md("""## 5. Backbone loaders and paired generators

BigGAN uses one fixed ImageNet class, StyleGAN2 one fixed initial identity, and
Stable Diffusion one fixed prompt. The latent directions and per-frame diffusion
seeds are identical across paired control conditions.
"""),
code("""MODELS={}

def load_biggan():
    if 'biggan' not in MODELS:
        from pytorch_pretrained_biggan import BigGAN
        MODELS['biggan']=BigGAN.from_pretrained('biggan-deep-128').cuda().eval()
    return MODELS['biggan']

def load_stylegan2():
    if 'stylegan2' not in MODELS:
        import dnnlib, legacy
        with dnnlib.util.open_url(STYLEGAN_URL) as f: MODELS['stylegan2']=legacy.load_network_pkl(f)['G_ema'].cuda().eval()
    return MODELS['stylegan2']

def load_sd():
    if 'sd' not in MODELS:
        from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
        model_id='stable-diffusion-v1-5/stable-diffusion-v1-5'
        text=StableDiffusionPipeline.from_pretrained(model_id,torch_dtype=torch.float16,safety_checker=None).to('cuda')
        img=StableDiffusionImg2ImgPipeline(**text.components).to('cuda'); MODELS['sd']=(text,img)
    return MODELS['sd']

def partial_stylegan(G,z):
    labels=torch.zeros((len(z),G.c_dim),device='cuda'); ws=G.mapping(z,labels,truncation_psi=TRUNCATION)
    x=img=None; w_idx=0
    for res in G.synthesis.block_resolutions:
        block=getattr(G.synthesis,f'b{res}'); block_ws=ws.narrow(1,w_idx,block.num_conv+block.num_torgb); w_idx+=block.num_conv
        x,img=block(x,img,block_ws,noise_mode='const')
        if res==STYLEGAN_RESOLUTION: break
    return img

def generate_biggan(control,seed):
    from pytorch_pretrained_biggan import one_hot_from_names
    model=load_biggan(); z=latent_walk(128,control,seed,MOTION_SCALE['biggan'])
    classes=one_hot_from_names(FIXED_BIGGAN_CLASS,batch_size=len(z)).astype('float32'); out=[]
    with torch.no_grad():
        for i in range(0,len(z),8):
            im=model(torch.from_numpy(z[i:i+8]).cuda(),torch.from_numpy(classes[i:i+8]).cuda(),TRUNCATION)
            out.append(((im+1)/2).clamp(0,1).permute(0,2,3,1).cpu().numpy())
    return np.concatenate(out),{'latent':z,'class_vectors':classes}

def generate_stylegan2(control,seed):
    G=load_stylegan2(); z=latent_walk(G.z_dim,control,seed,MOTION_SCALE['stylegan2']); out=[]
    with torch.no_grad():
        for i in range(0,len(z),4):
            im=partial_stylegan(G,torch.from_numpy(z[i:i+4]).cuda())
            out.append(((im+1)/2).clamp(0,1).permute(0,2,3,1).cpu().numpy())
    return np.concatenate(out),{'latent':z}

def generate_sd(control,seed):
    text,imgpipe=load_sd(); lo,hi=SD_STRENGTH
    scaled=np.clip(MOTION_SCALE['sd']*control,0,1); strengths=lo+(hi-lo)*np.r_[scaled[0],scaled[:-1]]
    frames=[]; init=None
    with torch.no_grad():
        for t in range(len(control)):
            generator=torch.Generator(device='cpu').manual_seed(seed+t)
            kw=dict(prompt=FIXED_PROMPT,negative_prompt=NEGATIVE_PROMPT,num_inference_steps=SD_STEPS,
                    guidance_scale=SD_CFG,generator=generator)
            if init is None: image=text(**kw).images[0]
            else: image=imgpipe(image=init,strength=float(strengths[t]),**kw).images[0]
            frames.append(np.asarray(image,dtype='float32')/255); init=image
    return np.stack(frames),{'strength':strengths.astype('float32'),'prompt':np.array(FIXED_PROMPT)}

GENERATORS={'biggan':generate_biggan,'stylegan2':generate_stylegan2,'sd':generate_sd}
print('Backbone functions ready.')
"""),
md("""## 6. Held-out adapter calibration

For a paper run, the notebook sweeps adapter settings on a deterministic
eight-transition trace that is not derived from any study song. It selects the
setting closest to the preregistered median LPIPS target of 0.02 and saves the
full candidate table. You may instead paste previously frozen values into the
configuration cell and set `RUN_CALIBRATION = False`.
"""),
code("""frozen_scales=dict(CALIBRATED_SCALES or {})
RUN_CALIBRATION = PAPER_RUN and (set(frozen_scales)<{'biggan','stylegan2'} or CALIBRATED_SD_INTERVAL is None)
CALIBRATION_TARGET = .02
CALIBRATION_RESULTS = {}

if RUN_CALIBRATION:
    cal_control=np.array([0,.15,.85,.30,1.0,.45,.70,.20],dtype='float32')
    for backbone in BACKBONES:
        candidates=[]
        if backbone=='sd':
            if CALIBRATED_SD_INTERVAL is not None:
                SD_STRENGTH=tuple(CALIBRATED_SD_INTERVAL); continue
            specs=[(.05,.20),(.10,.30),(.20,.50),(.30,.65)]
            old_interval=SD_STRENGTH
            for interval in specs:
                SD_STRENGTH=interval; frames,_=GENERATORS[backbone](cal_control,991)
                median=float(np.median(lpips_signal(frames)))
                candidates.append({'strength_min':interval[0],'strength_max':interval[1],
                                   'median_lpips':median,'absolute_error':abs(median-CALIBRATION_TARGET)})
            selected=min(candidates,key=lambda row:row['absolute_error'])
            SD_STRENGTH=(selected['strength_min'],selected['strength_max'])
            CALIBRATED_SD_INTERVAL=SD_STRENGTH
        else:
            if backbone in frozen_scales:
                MOTION_SCALE[backbone]=frozen_scales[backbone]; continue
            old_scale=MOTION_SCALE[backbone]
            for scale in (.5,1.,2.,4.):
                MOTION_SCALE[backbone]=scale; frames,_=GENERATORS[backbone](cal_control,991)
                median=float(np.median(lpips_signal(frames)))
                candidates.append({'motion_scale':scale,'median_lpips':median,
                                   'absolute_error':abs(median-CALIBRATION_TARGET)})
            selected=min(candidates,key=lambda row:row['absolute_error'])
            MOTION_SCALE[backbone]=selected['motion_scale']
            frozen_scales[backbone]=selected['motion_scale']
        CALIBRATION_RESULTS[backbone]={'candidates':candidates,'selected':selected}
        print(backbone,'selected',selected)
    CALIBRATED_SCALES=dict(frozen_scales)
    (OUTPUT_ROOT/'calibration.json').write_text(json.dumps(
        {'target_median_lpips':CALIBRATION_TARGET,'seed':991,'frames':8,
         'results':CALIBRATION_RESULTS},indent=2))

if PAPER_RUN:
    assert set(CALIBRATED_SCALES or {}) >= {'biggan','stylegan2'} and CALIBRATED_SD_INTERVAL is not None, 'Calibration is required.'
print('Motion scales:',MOTION_SCALE,'| SD interval:',SD_STRENGTH)
"""),
md("""## 7. Execute the selected matrix

Completed runs are detected through `metrics.json` and skipped, so this cell can
resume when `OUTPUT_ROOT` points to mounted Drive storage. Never mix smoke
outputs with paper results.
"""),
code("""for song_path in SELECTED_SONGS:
    song_id=Path(song_path).stem; features=audio_request(song_path); raw=features['raw_motion']
    for backbone in BACKBONES:
        for seed in SEEDS:
            for condition in CONDITIONS:
                run_dir=OUTPUT_ROOT/f'{song_id}_{backbone}_seed{seed}_{condition}'
                metric_file=run_dir/'metrics.json'
                if metric_file.exists():
                    RESULTS.append(json.loads(metric_file.read_text())); print('cached',run_dir.name); continue
                control=make_control(raw,condition,seed); print('running',run_dir.name)
                t0=time.time(); frames,adapter_meta=GENERATORS[backbone](control,seed); elapsed=time.time()-t0
                meta={**features,**adapter_meta,'motion_control':control,'control_condition':np.array(condition),'seed':np.array(seed)}
                result=save_run(run_dir,frames,meta,elapsed,backbone,song_id,seed,condition,song_path)
                RESULTS.append(result); print(' done:',{k:round(v,4) if isinstance(v,float) else v for k,v in result.items()})
                torch.cuda.empty_cache()
print('Matrix selection complete:',len(RESULTS),'sequences')
"""),
md("""## 8. Paired causal summary and export

Interpret aligned-minus-control contrasts within each song, seed, and backbone.
Smoke results validate execution only and must not be copied into the paper.
"""),
code("""df=pd.DataFrame(RESULTS)
display(df.sort_values(['name','song','seed','control_condition']).round(4))

controls=df[df.control_condition.isin(['shifted','shuffled','matched_random'])]
aligned=df[df.control_condition.eq('aligned')][['name','song','seed','motion_sync']].rename(columns={'motion_sync':'aligned_sync'})
control_mean=controls.groupby(['name','song','seed'],as_index=False).motion_sync.mean().rename(columns={'motion_sync':'control_sync'})
paired=aligned.merge(control_mean,on=['name','song','seed'],how='left')
paired['delta_sync']=paired.aligned_sync-paired.control_sync
display(paired.round(4))

df.to_csv(OUTPUT_ROOT/'metrics.csv',index=False); paired.to_csv(OUTPUT_ROOT/'paired_contrasts.csv',index=False)
!cd /content && zip -qr musicviz_causal_results.zip musicviz_causal
from google.colab import files
files.download('/content/musicviz_causal_results.zip')
print('Exported metrics, paired contrasts, metadata, frames, and videos.')
"""),
md("""## Validity checklist

- Use three licensed songs and all three seeds for reported results.
- Calibrate `MOTION_SCALE` on a held-out synthetic trace before the paper run.
- Preserve the same timestamps, prompt/class, initial state, and random seed
  across each aligned/control set.
- Report LPIPS-motion correspondence, lag curves, paired uncertainty, compute,
  and the blinded paired user study.
- Treat latent-step agreement as an implementation check, not research evidence.
""")]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "musicviz_causal_protocol.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"}},
    "nbformat": 4,
    "nbformat_minor": 5}

target = Path(__file__).with_name("musicviz_baselines.ipynb")
target.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {target} ({len(cells)} cells)")
