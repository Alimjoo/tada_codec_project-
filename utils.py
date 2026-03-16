import math

import torch


def _fft_device(audio: torch.Tensor) -> torch.device:
    if audio.device.type == "mps":
        return torch.device("cpu")
    return audio.device


def build_assignment_indicator(T: int, positions: torch.Tensor) -> torch.Tensor:
    """
    positions: [B, L]
    returns: [B, T, 1]
    """
    B, _ = positions.shape
    indicator = torch.zeros(B, T, 1, device=positions.device)
    clipped = positions.clamp(0, max(T - 1, 0))
    indicator.scatter_(1, clipped.unsqueeze(-1), 1.0)
    return indicator



def extract_aligned_tokens(frame_feats: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    frame_feats: [B, T, D]
    positions: [B, L]
    returns: [B, L, D]
    """
    B, T, D = frame_feats.shape
    idx = positions.clamp(0, max(T - 1, 0)).unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(frame_feats, dim=1, index=idx)



def scatter_tokens_to_frames(tokens: torch.Tensor, positions: torch.Tensor, T: int) -> torch.Tensor:
    """
    tokens: [B, L, D]
    positions: [B, L]
    returns: [B, T, D]
    """
    B, L, D = tokens.shape
    out = torch.zeros(B, T, D, device=tokens.device, dtype=tokens.dtype)
    idx = positions.clamp(0, max(T - 1, 0)).unsqueeze(-1).expand(-1, -1, D)
    out.scatter_(1, idx, tokens)
    return out



def log_mel_spectrogram(audio: torch.Tensor,
                        n_fft: int,
                        hop_length: int,
                        win_length: int,
                        n_mels: int) -> torch.Tensor:
    """
    Lightweight mel-ish spectrogram for training starter.
    Replace with torchaudio MelSpectrogram for production.
    audio: [B, T]
    returns: [B, n_mels, frames]
    """
    if audio.dim() != 2:
        raise ValueError(f"expected audio with shape [B, T], got {tuple(audio.shape)}")

    fft_device = _fft_device(audio)
    if fft_device != audio.device:
        audio = audio.to(fft_device)

    if audio.size(1) < win_length:
        pad = win_length - audio.size(1)
        audio = torch.nn.functional.pad(audio, (0, pad))

    window = torch.hann_window(win_length, device=audio.device, dtype=audio.dtype)
    frames = audio.unfold(dimension=1, size=win_length, step=hop_length)
    frames = frames * window.view(1, 1, win_length)

    if n_fft > win_length:
        fft_pad = n_fft - win_length
        frames = torch.nn.functional.pad(frames, (0, fft_pad))

    spec = torch.fft.rfft(frames, n=n_fft, dim=-1)
    mag = spec.abs().transpose(1, 2)  # [B, F, TT]

    freq_bins = mag.size(1)
    mel_basis = torch.linspace(0.0, 1.0, steps=n_mels, device=audio.device).unsqueeze(1)
    freqs = torch.linspace(0.0, 1.0, steps=freq_bins, device=audio.device).unsqueeze(0)
    mel_basis = 1.0 - (mel_basis - freqs).abs()
    mel_basis = mel_basis.clamp_min(0.0)
    mel_basis = mel_basis / (mel_basis.sum(dim=1, keepdim=True) + 1e-8)

    mel = torch.einsum("mf,bft->bmt", mel_basis, mag)
    return torch.log(mel.clamp_min(1e-5))



def approx_num_frames(audio_len: int, hop_length: int) -> int:
    return math.ceil(audio_len / hop_length)
