import argparse
from dataclasses import fields
from pathlib import Path

import torch
import torchaudio

from config import CodecConfig, resolve_device
from model import TADACodec


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_cfg = ckpt.get("config", CodecConfig())
    if isinstance(raw_cfg, dict):
        valid_fields = {field.name for field in fields(CodecConfig)}
        cfg = CodecConfig(**{key: value for key, value in raw_cfg.items() if key in valid_fields})
    else:
        cfg = raw_cfg
    cfg.device = resolve_device(device)

    model = TADACodec(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="test_outputs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    model, cfg, _ = load_model(args.ckpt, args.device)
    items = torch.load(args.data, map_location="cpu")

    if not items:
        raise ValueError("dataset is empty")
    if args.index < 0 or args.index >= len(items):
        raise IndexError(f"index {args.index} out of range for dataset of size {len(items)}")

    sample = items[args.index]
    audio = sample["audio"].unsqueeze(0).to(args.device)
    positions = sample["positions"].unsqueeze(0).to(args.device)
    audio_lens = torch.tensor([sample["audio"].numel()], device=args.device)
    text_lens = torch.tensor([sample["text_ids"].numel()], device=args.device)

    with torch.no_grad():
        out = model(audio, positions, audio_lens, text_lens)

    wav_hat = out["wav_hat"][0].detach().cpu()
    wav_ref = sample["audio"].detach().cpu()

    length = min(wav_ref.numel(), wav_hat.numel())
    wav_ref = wav_ref[:length].unsqueeze(0)
    wav_hat = wav_hat[:length].unsqueeze(0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir / f"sample_{args.index}_ref.wav"
    recon_path = out_dir / f"sample_{args.index}_recon.wav"
    meta_path = out_dir / f"sample_{args.index}_meta.txt"

    torchaudio.save(str(ref_path), wav_ref, sample_rate=cfg.sample_rate)
    torchaudio.save(str(recon_path), wav_hat, sample_rate=cfg.sample_rate)

    meta = [
        f"index: {args.index}",
        f"sample_rate: {cfg.sample_rate}",
        f"reference_text: {sample.get('reference_text', '')}",
        f"predicted_transcript: {sample.get('predicted_transcript', '')}",
        f"text_ids_len: {sample['text_ids'].numel()}",
        f"positions_len: {sample['positions'].numel()}",
        f"saved_ref: {ref_path}",
        f"saved_recon: {recon_path}",
    ]
    meta_path.write_text("\n".join(meta), encoding="utf-8")

    print(f"saved reference audio to {ref_path}")
    print(f"saved reconstruction audio to {recon_path}")
    print(f"saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
