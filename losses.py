import torch
import torch.nn.functional as F
from utils import log_mel_spectrogram, extract_aligned_tokens



def mel_loss_fn(wav_hat: torch.Tensor, wav: torch.Tensor, cfg) -> torch.Tensor:
    mel_hat = log_mel_spectrogram(wav_hat, cfg.n_fft, cfg.hop_length, cfg.win_length, cfg.n_mels)
    mel = log_mel_spectrogram(wav, cfg.n_fft, cfg.hop_length, cfg.win_length, cfg.n_mels)
    T = min(mel_hat.size(-1), mel.size(-1))
    return F.l1_loss(mel_hat[..., :T], mel[..., :T])



def kl_loss_fn(mu: torch.Tensor, floor: float = 0.5) -> torch.Tensor:
    kl = mu.pow(2).mean()
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
