"""
Extract embeddings from a manifest of paired (frame, audio) clips.
 
Manifest CSV columns: clip_id, label, frame_path, audio_path
(e.g. built from VGGSound: one middle frame per clip + its audio track).
 
Usage:
    python extract_embeddings.py --manifest manifest.csv --model dinov2-base --out dino_b.npz
    python extract_embeddings.py --manifest manifest.csv --model clap        --out clap.npz
 
Vision : dinov2-{small,base,large}, clip-{b32,b16,l14}, mae-base
Audio  : clap (language-supervised), ast (supervised AudioSet),
         wav2vec2-{base,large}, hubert-{base,large} (speech-oriented self-supervised)
For BEATs / AudioMAE use their official repos and save the same npz format:
    np.savez(out, X=embeddings, labels=labels)
"""
import argparse
import numpy as np, pandas as pd, torch
from PIL import Image
from tqdm import tqdm
import librosa
from transformers import (AutoModel, AutoImageProcessor, CLIPModel, CLIPProcessor,
                          ClapModel, ClapProcessor, AutoFeatureExtractor, ViTMAEModel)
 
VISION = {
    "dinov2-small": "facebook/dinov2-small", "dinov2-base": "facebook/dinov2-base",
    "dinov2-large": "facebook/dinov2-large",
    "clip-b32": "openai/clip-vit-base-patch32", "clip-b16": "openai/clip-vit-base-patch16",
    "clip-l14": "openai/clip-vit-large-patch14", "mae-base": "facebook/vit-mae-base",
}
AUDIO = {
    "clap": "laion/clap-htsat-unfused", "ast": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "wav2vec2-base": "facebook/wav2vec2-base", "wav2vec2-large": "facebook/wav2vec2-large",
    "hubert-base": "facebook/hubert-base-ls960", "hubert-large": "facebook/hubert-large-ll60k",
}
dev = "cuda" if torch.cuda.is_available() else "cpu"
 
 
@torch.no_grad()
def vision_embed(name, paths, bs=32):
    ckpt = VISION[name]
    if name.startswith("clip"):
        m, proc = CLIPModel.from_pretrained(ckpt).to(dev).eval(), CLIPProcessor.from_pretrained(ckpt)
        f = lambda x: m.get_image_features(pixel_values=x)
    else:
        proc = AutoImageProcessor.from_pretrained(ckpt)
        m = (ViTMAEModel.from_pretrained(ckpt, mask_ratio=0.0) if name.startswith("mae")
             else AutoModel.from_pretrained(ckpt)).to(dev).eval()
        f = lambda x: m(pixel_values=x).last_hidden_state.mean(1)  # mean-pooled tokens
    out = []
    for i in tqdm(range(0, len(paths), bs), desc=name):
        ims = [Image.open(p).convert("RGB") for p in paths[i:i + bs]]
        px = proc(images=ims, return_tensors="pt")["pixel_values"].to(dev)
        out.append(f(px).float().cpu().numpy())
    return np.concatenate(out)
 
 
@torch.no_grad()
def audio_embed(name, paths, bs=16):
    ckpt = AUDIO[name]
    sr = 48000 if name == "clap" else 16000
    if name == "clap":
        m, proc = ClapModel.from_pretrained(ckpt).to(dev).eval(), ClapProcessor.from_pretrained(ckpt)
        f = lambda b: m.get_audio_features(**b)
        mk = lambda w: proc(audios=w, sampling_rate=sr, return_tensors="pt")
    else:
        proc = AutoFeatureExtractor.from_pretrained(ckpt)
        m = AutoModel.from_pretrained(ckpt).to(dev).eval()
        f = lambda b: m(**b).last_hidden_state.mean(1)  # mean-pooled frames
        mk = lambda w: proc(w, sampling_rate=sr, return_tensors="pt", padding=True)
    out = []
    for i in tqdm(range(0, len(paths), bs), desc=name):
        wavs = [librosa.load(p, sr=sr, mono=True, duration=10)[0] for p in paths[i:i + bs]]
        b = {k: v.to(dev) for k, v in mk(wavs).items()}
        out.append(f(b).float().cpu().numpy())
    return np.concatenate(out)
 
 
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.manifest)
    if a.model in VISION:
        X = vision_embed(a.model, df.frame_path.tolist())
    elif a.model in AUDIO:
        X = audio_embed(a.model, df.audio_path.tolist())
    else:
        raise SystemExit(f"unknown model {a.model}")
    np.savez(a.out, X=X, labels=df.label.values, clip_id=df.clip_id.values)
    print("saved", a.out, X.shape)
 