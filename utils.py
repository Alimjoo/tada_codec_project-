import math

import torch


def _fft_device(audio: torch.Tensor) -> torch.device:
    if audio.device.type == "mps":
        return torch.device("cpu")
    return audio.device


def build_assignment_indicator(T: int, positions: torch.Tensor) -> torch.Tensor:
    bsz, _ = positions.shape
    indicator = torch.zeros(bsz, T, 1, device=positions.device)
    clipped = positions.clamp(0, max(T - 1, 0))
    indicator.scatter_(1, clipped.unsqueeze(-1), 1.0)
    return indicator


def lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    steps = torch.arange(max_len, device=lengths.device)
    return steps.unsqueeze(0) >= lengths.unsqueeze(1)


def audio_lengths_to_frame_lengths(audio_lens: torch.Tensor, hop_length: int) -> torch.Tensor:
    return ((audio_lens + hop_length - 1) // hop_length).long()


def conv_output_lengths(
    lengths: torch.Tensor,
    strides: tuple[int, ...],
    kernel_fn=None,
    padding_fn=None,
) -> torch.Tensor:
    out = lengths.clone()
    for stride in strides:
        kernel = kernel_fn(stride) if kernel_fn is not None else (stride * 2 + 1)
        padding = padding_fn(stride) if padding_fn is not None else (stride // 2 + 1)
        out = ((out + 2 * padding - kernel) // stride) + 1
    return out.long().clamp_min(1)


def build_encoder_attention_mask(positions: torch.Tensor, frame_lens: torch.Tensor) -> torch.Tensor:
    batch = positions.size(0)
    max_frames = int(frame_lens.max().item())
    mask = torch.ones(batch, max_frames, max_frames, device=positions.device, dtype=torch.bool)
    for b in range(batch):
        frame_len = int(frame_lens[b].item())
        if frame_len <= 0:
            continue
        valid_pos = positions[b]
        valid_pos = valid_pos[(valid_pos > 0) & (valid_pos < frame_len)]
        bounds = torch.cat(
            [
                torch.tensor([-1], device=positions.device, dtype=torch.long),
                valid_pos,
                torch.tensor([frame_len], device=positions.device, dtype=torch.long),
            ]
        )
        sample_mask = torch.ones(frame_len, frame_len, device=positions.device, dtype=torch.bool)
        for idx in range(1, bounds.numel() - 1):
            center = int(bounds[idx].item())
            left = int(bounds[idx - 1].item()) + 1
            right = int(bounds[idx + 1].item()) - 1
            if left <= right:
                sample_mask[center, left : right + 1] = False
            inner_left = max(left, center + 1)
            if inner_left <= right:
                sample_mask[inner_left : right + 1, inner_left : right + 1] = False
        sample_mask.fill_diagonal_(False)
        mask[b, :frame_len, :frame_len] = sample_mask
    return mask


def build_decoder_attention_mask(
    positions: torch.Tensor,
    frame_lens: torch.Tensor,
    context_blocks: int = 2,
) -> torch.Tensor:
    batch = positions.size(0)
    max_frames = int(frame_lens.max().item())
    mask = torch.ones(batch, max_frames, max_frames, device=positions.device, dtype=torch.bool)
    for b in range(batch):
        frame_len = int(frame_lens[b].item())
        if frame_len <= 0:
            continue
        valid_pos = positions[b]
        valid_pos = valid_pos[(valid_pos > 0) & (valid_pos < frame_len)]
        bounds = torch.cat(
            [
                torch.tensor([-1], device=positions.device, dtype=torch.long),
                valid_pos,
                torch.tensor([frame_len], device=positions.device, dtype=torch.long),
            ]
        )
        sample_mask = torch.ones(frame_len, frame_len, device=positions.device, dtype=torch.bool)
        for block_idx in range(bounds.numel() - 1):
            start = int(bounds[block_idx].item()) + 1
            end = int(bounds[block_idx + 1].item()) - 1
            if start > end:
                continue
            left_block = max(0, block_idx - context_blocks)
            allowed_start = int(bounds[left_block].item()) + 1
            sample_mask[start : end + 1, allowed_start : end + 1] = False
        sample_mask |= torch.triu(torch.ones_like(sample_mask), diagonal=1)
        sample_mask.fill_diagonal_(False)
        mask[b, :frame_len, :frame_len] = sample_mask
    return mask


def expand_attention_mask(attn_mask: torch.Tensor, n_heads: int) -> torch.Tensor:
    batch, tgt, src = attn_mask.shape
    return attn_mask.unsqueeze(1).expand(batch, n_heads, tgt, src).reshape(batch * n_heads, tgt, src)


def extract_aligned_tokens(frame_feats: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    _, T, D = frame_feats.shape
    idx = positions.clamp(0, max(T - 1, 0)).unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(frame_feats, dim=1, index=idx)


def scatter_tokens_to_frames(tokens: torch.Tensor, positions: torch.Tensor, T: int) -> torch.Tensor:
    B, _, D = tokens.shape
    out = torch.zeros(B, T, D, device=tokens.device, dtype=tokens.dtype)
    idx = positions.clamp(0, max(T - 1, 0)).unsqueeze(-1).expand(-1, -1, D)
    out.scatter_(1, idx, tokens)
    return out


def log_mel_spectrogram(
    audio: torch.Tensor,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
) -> torch.Tensor:
    if audio.dim() != 2:
        raise ValueError(f"expected audio with shape [B, T], got {tuple(audio.shape)}")

    fft_device = _fft_device(audio)
    if fft_device != audio.device:
        audio = audio.to(fft_device)

    if audio.size(1) < win_length:
        audio = torch.nn.functional.pad(audio, (0, win_length - audio.size(1)))

    window = torch.hann_window(win_length, device=audio.device, dtype=audio.dtype)
    frames = audio.unfold(dimension=1, size=win_length, step=hop_length)
    frames = frames * window.view(1, 1, win_length)
    if n_fft > win_length:
        frames = torch.nn.functional.pad(frames, (0, n_fft - win_length))

    spec = torch.fft.rfft(frames, n=n_fft, dim=-1)
    mag = spec.abs().transpose(1, 2)

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
