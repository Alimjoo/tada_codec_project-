import torch
import torch.nn.functional as F
from utils import log_mel_spectrogram, extract_aligned_tokens



def mel_loss_fn(wav_hat: torch.Tensor, wav: torch.Tensor, cfg) -> torch.Tensor:
    total = 0.0
    for n_fft, hop_length, win_length in cfg.mel_scales:
        mel_hat = log_mel_spectrogram(wav_hat, n_fft, hop_length, win_length, cfg.n_mels)
        mel = log_mel_spectrogram(wav, n_fft, hop_length, win_length, cfg.n_mels)
        t = min(mel_hat.size(-1), mel.size(-1))
        total = total + F.l1_loss(mel_hat[..., :t], mel[..., :t])
    return total / max(len(cfg.mel_scales), 1)



def kl_loss_fn(mu: torch.Tensor, logvar: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
    kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
    kl = kl.mean()
    return torch.clamp(kl, min=floor)



def semantic_loss_fn(sem_logits: torch.Tensor, positions: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
    gathered = extract_aligned_tokens(sem_logits, positions)  # [B, L, V]
    B, L, V = gathered.shape
    return F.cross_entropy(gathered.reshape(B * L, V), text_ids.reshape(B * L), ignore_index=0)



def discriminator_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    return ((real_scores - 1.0) ** 2).mean() + (fake_scores ** 2).mean()



def generator_adv_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    return ((fake_scores - 1.0) ** 2).mean()



def feature_matching_loss(real_feats, fake_feats) -> torch.Tensor:
    total = 0.0
    for r, f in zip(real_feats, fake_feats):
        total = total + F.l1_loss(f, r.detach())
    return total
