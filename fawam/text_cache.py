"""Cached UMT5 text embeddings (the server's conditioning: 512 tokens, zero-padded, bf16).

  python -m fawam.text_cache build --model_path lingbot-va/checkpoints/lingbot-va-posttrain-robotwin <episode roots...>

writes artifacts/text_emb/<sha1>.pt for every instruction/prompt found (+ the empty string for CFG dropout).
"""
import argparse
import hashlib
from pathlib import Path

import torch

from fawam.paths import ROOT

DEFAULT_DIR = ROOT / "artifacts" / "text_emb"


def key(text):
    return hashlib.sha1(text.encode()).hexdigest()


class TextCache:
    def __init__(self, cache_dir=DEFAULT_DIR):
        self.dir = Path(cache_dir)
        self.mem = {}

    def get(self, text):
        k = key(text)
        if k not in self.mem:
            p = self.dir / f"{k}.pt"
            if not p.exists():
                raise FileNotFoundError(f"no cached embedding for {text!r} ({p}); run fawam.text_cache build")
            self.mem[k] = torch.load(p, map_location="cpu", weights_only=True)
        return self.mem[k]

    def missing(self, texts):
        return sorted({t for t in texts if not (self.dir / f"{key(t)}.pt").exists()})


@torch.no_grad()
def encode(texts, model_path, cache_dir=DEFAULT_DIR, device="cuda", batch=8):
    from fawam.paths import add_lingbot_to_path
    add_lingbot_to_path()
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    from modules.utils import load_text_encoder, load_tokenizer
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer(str(Path(model_path) / "tokenizer"))
    enc = load_text_encoder(str(Path(model_path) / "text_encoder"), torch_dtype=torch.bfloat16, torch_device=device)
    texts = list(texts)
    for i in range(0, len(texts), batch):
        part = texts[i:i + batch]
        ti = tok([prompt_clean(t) for t in part], padding="max_length", max_length=512, truncation=True,
                 add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        out = enc(ti.input_ids.to(device), ti.attention_mask.to(device)).last_hidden_state
        lens = ti.attention_mask.sum(1)
        for t, e, n in zip(part, out, lens):
            emb = torch.zeros_like(e)
            emb[:n] = e[:n]
            torch.save(emb.to(torch.bfloat16).cpu(), cache_dir / f"{key(t)}.pt")
    return len(texts)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("roots", nargs="+")
    b.add_argument("--model_path", required=True)
    b.add_argument("--cache_dir", default=str(DEFAULT_DIR))
    b.add_argument("--extra", nargs="*", default=[], help="additional texts (e.g. corrective prompts)")
    a = ap.parse_args()
    import h5py
    from fawam.latent_data import find_episodes
    texts = {""} | set(a.extra)
    for e in find_episodes(a.roots):
        with h5py.File(e / "episode.h5", "r") as f:
            for k in ("instruction", "prompt"):
                if k in f.attrs:
                    texts.add(str(f.attrs[k]))
        for ana in e.glob("analysis*.json"):  # Learn-from-Failure: correction prompts
            import json
            c = json.loads(ana.read_text()).get("correction")
            if c:
                texts.add(c)
    todo = TextCache(a.cache_dir).missing(texts)
    print(f"{len(texts)} texts, {len(todo)} to encode")
    if todo:
        encode(todo, a.model_path, a.cache_dir)


if __name__ == "__main__":
    main()
