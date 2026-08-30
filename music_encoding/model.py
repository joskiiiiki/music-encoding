import torch as tc
from torch import nn


class BarlowTwinsLoss:
    def __init__(self, lambd: float = 2e-2):
        self.lambd = lambd

    def __call__(
        self,
        z_a: tc.Tensor,
        z_b: tc.Tensor,
    ) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor]:
        B, D = z_a.shape
        z_a = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-6)
        z_b = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-6)
        c = (z_a.T @ z_b) / B
        diag = tc.diagonal(c)
        on_diag = (diag - 1).pow(2).sum()
        off_diag = (
            c.pow(2).sum() - diag.pow(2).sum()
        )  # avoids the extra D×D diag_embed tensor
        return on_diag + self.lambd * off_diag, on_diag.detach(), off_diag.detach()


class SiameseEncoderBT(nn.Module):
    def __init__(
        self,
        embed_dims: int = 128,
        proj_dims: int = 8192,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten(),
        )
        # single conv path: 128 ch × 2×2 (AdaptiveAvgPool2d(2)) = 512 features.
        # No early-feature concat — the embed takes just the conv stack's output.
        self.embed = nn.Sequential(
            nn.Linear(128 * 2 * 2, embed_dims * 2),
            nn.SiLU(),
            nn.Linear(embed_dims * 2, embed_dims),
            nn.SiLU(),
            nn.Linear(embed_dims, embed_dims),
        )

        self.projector = nn.Sequential(
            nn.Linear(embed_dims, proj_dims),
            nn.BatchNorm1d(proj_dims),
            nn.SiLU(),
            nn.Linear(proj_dims, proj_dims),
            nn.BatchNorm1d(proj_dims),
            nn.SiLU(),
            nn.Linear(proj_dims, proj_dims),
        )

    def forward(self, spec: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
        # expects a (B, 1, n_mels, T) log-mel spectrogram, already computed and
        # augmented upstream (see LogMelSpectrogram in twin_dataset.py) — the
        # conv stack onward.
        x = self.conv(spec)
        x = x.view(x.size(0), -1)
        emb = self.embed(x)
        z = self.projector(emb)
        return emb, z

    def forward_pair(
        self, spec_a: tc.Tensor, spec_b: tc.Tensor
    ) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor]:
        spec = tc.cat([spec_a, spec_b], dim=0)
        emb, z = self.forward(spec)
        emb_a, emb_b = emb.chunk(2, dim=0)
        z_a, z_b = z.chunk(2, dim=0)
        return emb_a, emb_b, z_a, z_b
