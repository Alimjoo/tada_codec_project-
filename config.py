from dataclasses import dataclass

import torch


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class CodecConfig:
    sample_rate: int = 16000
    hop_length: int = 320
    n_fft: int = 1024
    win_length: int = 1024
    n_mels: int = 80
    vocab_size: int = 32000

    hidden_dim: int = 256
    latent_dim: int = 128
    n_layers: int = 4
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1

    lr_g: float = 2e-4
    lr_d: float = 2e-4
    grad_clip: float = 1.0
    kl_floor: float = 0.5

    mel_weight: float = 45.0
    sem_weight: float = 1.0
    kl_weight: float = 0.1
    gen_weight: float = 1.0
    fm_weight: float = 10.0

    device: str = get_best_device()
