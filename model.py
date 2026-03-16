import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, stride=5, padding=3),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv1d(256, hidden_dim, kernel_size=5, stride=4, padding=2),
            nn.GELU(),
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio.unsqueeze(1)
        x = self.net(x)
        return x.transpose(1, 2)


class ConvDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim, 256, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=4, padding=2),
            nn.GELU(),
            nn.ConvTranspose1d(128, 64, kernel_size=10, stride=5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.transpose(1, 2)
        x = self.net(x)
        return x.squeeze(1)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x)


class TADACodec(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ConvEncoder(cfg.hidden_dim)
        self.backbone = TransformerStack(
            dim=cfg.hidden_dim,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
        )
        self.mu_proj = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.latent_to_hidden = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.sem_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size)
        self.decoder = ConvDecoder(cfg.hidden_dim)

    def forward(self, audio: torch.Tensor, positions: torch.Tensor) -> dict:
        del positions
        frames = self.encoder(audio)
        hidden = self.backbone(frames)
        mu = self.mu_proj(hidden)
        hidden_for_decode = self.latent_to_hidden(mu)
        wav_hat = self.decoder(hidden_for_decode)
        sem_logits = self.sem_head(hidden)
        return {
            "mu": mu,
            "sem_logits": sem_logits,
            "wav_hat": wav_hat,
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
        scores = self.head(x)
        return scores, feats
