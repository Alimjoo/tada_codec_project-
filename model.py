import torch
import torch.nn as nn

from utils import (
    build_assignment_indicator,
    build_decoder_attention_mask,
    build_encoder_attention_mask,
    conv_output_lengths,
    expand_attention_mask,
    extract_aligned_tokens,
    lengths_to_mask,
    scatter_tokens_to_frames,
)


class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int, strides: tuple[int, ...]):
        super().__init__()
        channels = [1, 64, 128, 256, hidden_dim]
        layers = []
        for idx, stride in enumerate(strides):
            layers.extend(
                [
                    nn.Conv1d(
                        channels[idx],
                        channels[idx + 1],
                        kernel_size=stride * 2 + 1,
                        stride=stride,
                        padding=stride // 2 + 1,
                    ),
                    nn.GELU(),
                ]
            )
        self.net = nn.Sequential(*layers)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.net(audio.unsqueeze(1)).transpose(1, 2)


class ConvDecoder(nn.Module):
    def __init__(self, hidden_dim: int, strides: tuple[int, ...]):
        super().__init__()
        channels = [hidden_dim, 256, 128, 64, 32]
        layers = []
        for idx, stride in enumerate(strides):
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        channels[idx],
                        channels[idx + 1],
                        kernel_size=stride * 2,
                        stride=stride,
                        padding=stride // 2,
                    ),
                    nn.GELU(),
                ]
            )
        layers.extend(
            [
                nn.Conv1d(channels[-1], 1, kernel_size=7, padding=3),
                nn.Tanh(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.net(frames.transpose(1, 2)).squeeze(1)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(
            h,
            h,
            h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class TransformerStack(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(dim, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return self.ln_f(x)


class TADACodec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg.hidden_dim, cfg.encoder_strides)
        self.frame_input = nn.Linear(cfg.hidden_dim + 1, cfg.hidden_dim)
        self.encoder_backbone = TransformerStack(
            dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
        )
        self.mu_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.logvar_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.token_to_hidden = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.decoder_backbone = TransformerStack(
            dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
        )
        self.sem_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        self.decoder = ConvDecoder(cfg.hidden_dim, cfg.decoder_strides)

    def _sample(self, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        std = torch.exp(0.5 * logvar) + self.cfg.latent_std_bias
        if not self.training:
            return mu, std
        noise = torch.randn_like(std) * self.cfg.latent_noise_scale
        return mu + std * noise, std

    def forward(
        self,
        audio: torch.Tensor,
        positions: torch.Tensor,
        audio_lens: torch.Tensor,
        text_lens: torch.Tensor,
    ) -> dict:
        del text_lens
        frame_lens = conv_output_lengths(audio_lens, self.cfg.encoder_strides)
        encoded_frames = self.encoder(audio)
        max_frames = encoded_frames.size(1)

        indicator = build_assignment_indicator(max_frames, positions).to(encoded_frames.dtype)
        encoder_in = self.frame_input(torch.cat([encoded_frames, indicator], dim=-1))
        frame_pad_mask = lengths_to_mask(frame_lens, max_frames)
        encoder_mask = expand_attention_mask(
            build_encoder_attention_mask(positions, frame_lens),
            self.cfg.n_heads,
        )
        encoder_hidden = self.encoder_backbone(
            encoder_in,
            attn_mask=encoder_mask,
            key_padding_mask=frame_pad_mask,
        )

        token_hidden = extract_aligned_tokens(encoder_hidden, positions)
        mu = self.mu_proj(token_hidden)
        logvar = self.logvar_proj(token_hidden).clamp(max=self.cfg.kl_max_logvar)
        sampled_tokens, std = self._sample(mu, logvar)
        sparse_frames = scatter_tokens_to_frames(
            self.token_to_hidden(sampled_tokens),
            positions,
            max_frames,
        )

        decoder_mask = expand_attention_mask(
            build_decoder_attention_mask(
                positions,
                frame_lens,
                context_blocks=self.cfg.decoder_context_blocks,
            ),
            self.cfg.n_heads,
        )
        decoder_hidden = self.decoder_backbone(
            sparse_frames + 0.25 * encoder_hidden,
            attn_mask=decoder_mask,
            key_padding_mask=frame_pad_mask,
        )
        wav_hat = self.decoder(decoder_hidden)
        sem_logits = self.sem_head(decoder_hidden)
        return {
            "mu": mu,
            "logvar": logvar,
            "std": std,
            "sem_logits": sem_logits,
            "wav_hat": wav_hat,
            "frame_lens": frame_lens,
        }


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=15, stride=stride, padding=7),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleWaveDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DiscriminatorBlock(1, 32, 2),
                DiscriminatorBlock(32, 64, 2),
                DiscriminatorBlock(64, 128, 2),
                DiscriminatorBlock(128, 256, 2),
            ]
        )
        self.head = nn.Conv1d(256, 1, kernel_size=3, padding=1)

    def forward(self, wav: torch.Tensor):
        x = wav.unsqueeze(1)
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)
        return self.head(x), feats
