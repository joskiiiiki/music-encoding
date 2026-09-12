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
        # compute the BT statistic in fp32 regardless of the autocast dtype: it
        # is a batch-normalized cross-correlation whose off-diagonal sum of
        # squares must not be accumulated in fp16/bf16. Trivial cost here
        # (D^2 ~ 1M) next to the conv work upstream.
        z_a = z_a.float()
        z_b = z_b.float()
        z_a = (z_a - z_a.mean(0)) / (z_a.std(0) + 1e-6)
        z_b = (z_b - z_b.mean(0)) / (z_b.std(0) + 1e-6)
        c = (z_a.T @ z_b) / B
        diag = tc.diagonal(c)
        on_diag = (diag - 1).pow(2).sum()
        off_diag = (
            c.pow(2).sum() - diag.pow(2).sum()
        )  # avoids the extra D×D diag_embed tensor
        return on_diag + self.lambd * off_diag, on_diag.detach(), off_diag.detach()


class InfoNCELoss:
    """NT-Xent / SimCLR-style contrastive term — the inter-sample repulsion BT lacks.

    Barlow Twins pulls two views of the SAME track together (on_diag) and
    decorrelates DIMENSIONS (off_diag). It has no term that pushes *different*
    tracks apart, which is measurable: unrelated windows sit at cosine ~0.83 and the
    embedding's uniformity is ~3.3 nats worse than isotropic, so a window's nearest
    neighbour is its own track only ~0.44 of the time (see test_retrieval.py).

    This adds that term. Each view is treated as its own class, so every other
    window in the batch — 2B-2 of them — is a negative:

        L = -log( exp(sim(z_i, z_pos)/τ) / Σ_{j≠i} exp(sim(z_i, z_j)/τ) )

    Applied on the projector output `z` (SimCLR/BT convention: the projector is the
    representation the objective shapes; `emb` is what gets evaluated).

    No false negatives: the dataset draws one window-pair per track per epoch, so a
    shuffled batch of B tracks never contains a second window of the same track.
    (A future change to the pairing — multi-crop, or two pairs per track — would
    break this and require a mask.)

    Returns a scalar; callers weight it against BarlowTwinsLoss themselves. Note the
    scales are very different — BT totals ~640 while this is ~ln(2B-1) ≈ 6.6 at
    initialisation — so the weight is not a small fraction.
    """

    def __init__(self, temperature: float = 0.1) -> None:
        self.temperature = temperature

    def __call__(self, z_a: tc.Tensor, z_b: tc.Tensor) -> tc.Tensor:
        # fp32 upcast: this is a softmax over 2B logits, so accumulating it in
        # fp16/bf16 would lose precision for no gain (2B x 2B is ~590k entries)
        z_a = z_a.float()
        z_b = z_b.float()
        B = z_a.shape[0]
        z = tc.nn.functional.normalize(tc.cat([z_a, z_b], dim=0), dim=1)

        sim = (z @ z.T) / self.temperature
        # a row's own logit is not a candidate; the positive is at +B (and -B for
        # the b-views), which (arange + B) % 2B maps correctly in both halves
        sim.fill_diagonal_(float("-inf"))
        targets = (tc.arange(2 * B, device=z.device) + B) % (2 * B)
        return tc.nn.functional.cross_entropy(sim, targets)


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
