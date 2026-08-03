import torch as tc
import torchaudio as ta
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
        sr: int = 22050,
        n_mels: int = 64,
        embed_dims: int = 128,
        proj_dims: int = 8192,
    ) -> None:
        super().__init__()
        self.mel = ta.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=1024, hop_length=256, n_mels=n_mels
        )
        self.to_db = ta.transforms.AmplitudeToDB()

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
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed = nn.Linear(128, embed_dims)
        self.projector = nn.Sequential(
            nn.Linear(embed_dims, proj_dims),
            nn.BatchNorm1d(proj_dims),
            nn.SiLU(),
            nn.Linear(proj_dims, proj_dims),
            nn.BatchNorm1d(proj_dims),
            nn.SiLU(),
            nn.Linear(proj_dims, proj_dims),
        )

    def forward(self, wav: tc.Tensor) -> tuple[tc.Tensor, tc.Tensor]:
        x = self.to_db(self.mel(wav)).unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        emb = self.embed(x)
        z = self.projector(emb)
        return emb, z

    def forward_pair(
        self, wav_a: tc.Tensor, wav_b: tc.Tensor
    ) -> tuple[tc.Tensor, tc.Tensor, tc.Tensor, tc.Tensor]:
        wav = tc.cat([wav_a, wav_b], dim=0)
        emb, z = self.forward(wav)
        emb_a, emb_b = emb.chunk(2, dim=0)
        z_a, z_b = z.chunk(2, dim=0)
        return emb_a, emb_b, z_a, z_b
