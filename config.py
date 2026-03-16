from dataclasses import dataclass

import torch


def get_best_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_device(requested: str | None = None) -> str:
    if requested is None or requested == "auto":
        return get_best_device()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "cuda"
    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise RuntimeError("MPS was requested but is not available")
        return "mps"
    if requested == "cpu":
        return "cpu"
    raise ValueError(f"unsupported device: {requested}")


@dataclass
class CodecConfig:
    sample_rate: int = 16000
    hop_length: int = 320
    n_fft: int = 1024
    win_length: int = 1024
    n_mels: int = 80
    vocab_size: int = 32000

    hidden_dim: int = 384
    latent_dim: int = 192
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    encoder_strides: tuple[int, ...] = (4, 4, 4, 5)
    decoder_strides: tuple[int, ...] = (5, 4, 4, 4)
    latent_std_bias: float = 0.5
    latent_noise_scale: float = 1.0
    decoder_context_blocks: int = 2

    lr_g: float = 2e-4
    lr_d: float = 2e-4
    grad_clip: float = 1.0
    kl_floor: float = 0.5
    kl_max_logvar: float = 2.0

    mel_weight: float = 45.0
    sem_weight: float = 1.0
    kl_weight: float = 0.1
    gen_weight: float = 1.0
    fm_weight: float = 10.0
    mel_scales: tuple[tuple[int, int, int], ...] = (
        (512, 160, 512),
        (1024, 320, 1024),
        (2048, 640, 2048),
    )

    device: str = get_best_device()
