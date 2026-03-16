import argparse
import os
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CodecConfig, resolve_device
from dataset import AlignedSpeechDataset, collate_fn
from losses import (
    discriminator_loss,
    feature_matching_loss,
    generator_adv_loss,
    kl_loss_fn,
    mel_loss_fn,
    semantic_loss_fn,
)
from model import SimpleWaveDiscriminator, TADACodec


def get_autocast_context(device: str, enabled: bool, amp_dtype: str):
    if not enabled or device != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def train_step(model, disc, batch, opt_g, opt_d, scaler_g, scaler_d, cfg, use_gan: bool):
    audio = batch["audio"].to(cfg.device, dtype=torch.float32)
    audio_lens = batch["audio_lens"].to(cfg.device)
    text_ids = batch["text_ids"].to(cfg.device)
    text_lens = batch["text_lens"].to(cfg.device)
    positions = batch["positions"].to(cfg.device)

    with get_autocast_context(cfg.device, cfg.use_amp, cfg.amp_dtype):
        out = model(audio, positions, audio_lens, text_lens)
        wav_hat = out["wav_hat"]

        t = min(audio.size(1), wav_hat.size(1))
        audio_crop = audio[:, :t]
        wav_hat_crop = wav_hat[:, :t]

        loss_mel = mel_loss_fn(wav_hat_crop, audio_crop, cfg)
        loss_kl = kl_loss_fn(out["mu"], out["logvar"], cfg.kl_floor)
        loss_sem = semantic_loss_fn(out["sem_logits"], positions, text_ids)

        loss_adv = torch.tensor(0.0, device=cfg.device)
        loss_fm = torch.tensor(0.0, device=cfg.device)

        if use_gan:
            fake_scores, fake_feats = disc(wav_hat_crop)
            with torch.no_grad():
                real_scores, real_feats = disc(audio_crop)
            loss_adv = generator_adv_loss(fake_scores)
            loss_fm = feature_matching_loss(real_feats, fake_feats)

        loss_g = (
            cfg.mel_weight * loss_mel
            + cfg.sem_weight * loss_sem
            + cfg.kl_weight * loss_kl
            + cfg.gen_weight * loss_adv
            + cfg.fm_weight * loss_fm
        )

    opt_g.zero_grad(set_to_none=True)
    if scaler_g is not None:
        scaler_g.scale(loss_g).backward()
        scaler_g.unscale_(opt_g)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler_g.step(opt_g)
        scaler_g.update()
    else:
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt_g.step()

    loss_d = torch.tensor(0.0, device=cfg.device)
    if use_gan:
        with torch.no_grad():
            detached_fake = model(audio, positions, audio_lens, text_lens)["wav_hat"][:, :t]
        with get_autocast_context(cfg.device, cfg.use_amp, cfg.amp_dtype):
            real_scores, _ = disc(audio_crop)
            fake_scores, _ = disc(detached_fake)
            loss_d = discriminator_loss(real_scores, fake_scores)
        opt_d.zero_grad(set_to_none=True)
        if scaler_d is not None:
            scaler_d.scale(loss_d).backward()
            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip)
            scaler_d.step(opt_d)
            scaler_d.update()
        else:
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), cfg.grad_clip)
            opt_d.step()

    return {
        "loss_g": float(loss_g.item()),
        "loss_d": float(loss_d.item()),
        "mel": float(loss_mel.item()),
        "kl": float(loss_kl.item()),
        "sem": float(loss_sem.item()),
        "adv": float(loss_adv.item()),
        "fm": float(loss_fm.item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--use-gan", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, choices=("fp16", "bf16"), default="bf16")
    args = parser.parse_args()

    cfg = CodecConfig()
    cfg.device = resolve_device(args.device)
    cfg.use_amp = bool(args.amp and cfg.device == "cuda")
    cfg.amp_dtype = args.amp_dtype
    print("using device:", cfg.device)
    if cfg.use_amp:
        print("using amp:", cfg.amp_dtype)

    items = torch.load(args.data, map_location="cpu")
    inferred_vocab_size = max(
        int(item["text_ids"].max().item()) for item in items if item["text_ids"].numel() > 0
    ) + 1
    cfg.vocab_size = max(cfg.vocab_size, inferred_vocab_size)

    ds = AlignedSpeechDataset(items)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = TADACodec(cfg).to(cfg.device)
    disc = SimpleWaveDiscriminator().to(cfg.device) if args.use_gan else None

    opt_g = torch.optim.AdamW(model.parameters(), lr=cfg.lr_g, betas=(0.8, 0.99))
    opt_d = (
        torch.optim.AdamW(disc.parameters(), lr=cfg.lr_d, betas=(0.8, 0.99))
        if args.use_gan
        else None
    )
    scaler_g = torch.amp.GradScaler("cuda") if cfg.use_amp else None
    scaler_d = torch.amp.GradScaler("cuda") if cfg.use_amp and args.use_gan else None

    os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        if disc is not None:
            disc.train()

        running = {
            "loss_g": 0.0,
            "loss_d": 0.0,
            "mel": 0.0,
            "kl": 0.0,
            "sem": 0.0,
            "adv": 0.0,
            "fm": 0.0,
        }

        pbar = tqdm(dl, desc=f"epoch {epoch}")
        count = 0
        for batch in pbar:
            stats = train_step(
                model,
                disc,
                batch,
                opt_g,
                opt_d,
                scaler_g,
                scaler_d,
                cfg,
                args.use_gan,
            )
            count += 1
            for key in running:
                running[key] += stats[key]

            avg = {key: running[key] / count for key in running}
            postfix = {
                "loss_g": f"{avg['loss_g']:.3f}",
                "mel": f"{avg['mel']:.3f}",
                "sem": f"{avg['sem']:.3f}",
            }
            if args.use_gan:
                postfix["loss_d"] = f"{avg['loss_d']:.3f}"
            pbar.set_postfix(postfix)

        ckpt = {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "epoch": epoch,
        }
        if args.use_gan:
            ckpt["disc"] = disc.state_dict()

        path = os.path.join(args.save_dir, f"codec_epoch_{epoch}.pt")
        torch.save(ckpt, path)
        print("saved", path)


if __name__ == "__main__":
    main()
