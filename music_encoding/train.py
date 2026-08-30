import argparse
import csv
import math
import os
import pathlib
import sys
from collections.abc import Callable
from datetime import datetime

import torch as tc
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from music_encoding.augmenter import AudioAugmenter, Mixup, SpectrogramAugmenter
from music_encoding.model import BarlowTwinsLoss, SiameseEncoderBT
from music_encoding.mtg import MTGJamendoBase
from music_encoding.twin_dataset import FMAPairDataset, collate_pairs


def train(
    model: SiameseEncoderBT,
    dl: DataLoader,
    optimizer: Optimizer,
    scheduler: tc.optim.lr_scheduler.ReduceLROnPlateau,
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tuple[tc.Tensor, tc.Tensor, tc.Tensor]],
    mixup: Mixup | None = None,
    grad_accum: int = 1,
    epochs: int = 300,
    device: str = "cuda",
    start_epoch: int = 0,
    log_path: pathlib.Path | None = None,
    checkpoint_path: os.PathLike | None = None,
):

    checkpoint_path = (
        pathlib.Path("checkpoints")
        if checkpoint_path is None
        else pathlib.Path(checkpoint_path)
    )
    checkpoint_path.mkdir(exist_ok=True, parents=True)

    # projection dim D: off_diag sums D*(D-1) squared correlations, so its raw
    # magnitude scales with D². Also expose the normalized mean-squared
    # off-diagonal correlation (off_diag / (D*(D-1))), which has a ~1/B noise
    # floor and is comparable across runs with different D and batch size.
    D = model.projector[-1].out_features

    # open once, append per epoch — avoids reopening/rewriting the file every iteration
    log_file = None
    csv_writer = None
    if log_path is not None:
        is_new_file = not log_path.exists()
        log_file = open(log_path, "a", newline="")
        csv_writer = csv.writer(log_file)
        if is_new_file:
            csv_writer.writerow(
                ["epoch", "lr", "loss", "on_diag", "off_diag", "off_diag_norm"]
            )

    print(
        f"[train] starting: epochs={start_epoch}..{start_epoch + epochs - 1}, "
        f"batches/epoch={len(dl)}, grad_accum={grad_accum}, device={device}, "
        f"checkpoints -> {checkpoint_path}",
        flush=True,
    )

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            total_loss = 0.0
            total_on_diag = 0.0
            total_off_diag = 0.0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch {epoch}/{start_epoch + epochs - 1} (lr={lr}) ...",
                flush=True,
            )
            steps = 0  # completed optimizer steps (full accumulation windows)
            for batch_idx, (spec_a, spec_b) in enumerate(dl):
                print(f"[train] batch {batch_idx}")
                spec_a = spec_a.to(device)
                spec_b = spec_b.to(device)

                if mixup is not None:
                    # in-batch mixing; each side gets a different partner.
                    # The dataset already did waveform aug + mel + RRC, so Mixup
                    # now blends the resulting spectrograms.
                    spec_a = mixup(spec_a)
                    spec_b = mixup(spec_b)

                # zero gradients once per accumulation window, not per micro-batch
                if batch_idx % grad_accum == 0:
                    optimizer.zero_grad()

                with tc.autocast(device_type="cuda", dtype=tc.bfloat16):
                    emb_a, emb_b, z_a, z_b = model.forward_pair(spec_a, spec_b)
                    loss, on_diag, off_diag = loss_fn(z_a, z_b)

                # scale by 1/grad_accum so the summed gradient matches one batch
                # of size batch_size * grad_accum (logged loss stays unscaled)
                (loss / grad_accum).backward()
                total_loss += loss.item()
                total_on_diag += on_diag.item()
                total_off_diag += off_diag.item()

                if (batch_idx + 1) % grad_accum == 0:
                    optimizer.step()
                    steps += 1
                    if steps % 50 == 0:
                        print(
                            f"[train] epoch {epoch} step {steps} (batch {batch_idx}/{len(dl)}) "
                            f"loss={loss.item():.4f}",
                            flush=True,
                        )

            # flush a trailing partial accumulation window — drop_last makes
            # len(dl) not necessarily a multiple of grad_accum
            if len(dl) % grad_accum != 0:
                optimizer.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(dl)
            scheduler.step(avg_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            on_diag_val = total_on_diag / len(dl)
            off_diag_val = total_off_diag / len(dl)
            # mean-squared off-diagonal correlation, normalized by the D*(D-1) terms
            off_diag_norm = off_diag_val / (D * (D - 1))
            off_diag_mean_abs = math.sqrt(off_diag_norm)  # mean |c_ij|

            print(
                f"epoch: {epoch}",
                f"lr: {current_lr}",
                f"loss: {avg_loss}",
                f"on_diag: {on_diag_val}",
                f"off_diag: {off_diag_val}",
                f"off_diag_norm: {off_diag_norm}",
                f"off_diag_mean_abs: {off_diag_mean_abs:.4f}",
                sep=", ",
            )

            if csv_writer is not None and log_file is not None:
                csv_writer.writerow(
                    [epoch, current_lr, avg_loss, on_diag_val, off_diag_val, off_diag_norm]
                )
                log_file.flush()  # ensure it's on disk even if the run is killed mid-training

            tc.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                },
                checkpoint_path / f"checkpoint_e{epoch}.pt",
            )
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--resume", default=None, type=pathlib.Path)
    parser.add_argument("-e", "--epochs", default=100, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--grad-accum", default=1, type=int)
    parser.add_argument("--prefetch", default=2, type=int)
    parser.add_argument("--cache-dir", default=pathlib.Path("cached_mtg"), type=pathlib.Path)
    parser.add_argument(
        "--mtg-data", required=True, type=pathlib.Path,
        help="MTG-Jamendo data dir (contains autotagging.tsv, raw.meta.tsv)",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data)",
    )
    parser.add_argument(
        "--min-window-rms", default=0.03, type=float,
        help="re-draw windows quieter than this RMS (near-silence); 0 disables",
    )
    parser.add_argument("--lambd", default=5e-2, type=float)

    args = vars(parser.parse_args(sys.argv[1:]))
    resume: pathlib.Path | None = args["resume"]
    epochs: int = args["epochs"]

    print(f"[init] device={device} resume={resume} epochs={epochs}", flush=True)

    print("[init] loading checkpoint...", flush=True)
    checkpoint = (
        tc.load(resume, map_location=device)
        if resume is not None and resume.is_file()
        else None
    )
    print(f"[init] checkpoint loaded: {resume if checkpoint else 'none'}", flush=True)

    print("[init] loading MTG-Jamendo base dataset...", flush=True)
    base_ds = MTGJamendoBase(args["mtg_data"], audio_root=args["audio_root"])
    print(f"[init] base dataset loaded: {len(base_ds)} tracks", flush=True)

    print("[init] creating augmenters (waveform audio + spectrogram RRC)...", flush=True)
    augmenter = AudioAugmenter()
    spectrogram_augmenter = SpectrogramAugmenter()

    print("[init] building pair dataset (caching resampled tracks)...", flush=True)
    ds = FMAPairDataset(
        base_ds,
        cache_dir=args["cache_dir"],
        augmenter=augmenter,
        spectrogram_augmenter=spectrogram_augmenter,
        min_window_rms=args["min_window_rms"],
    )
    print(f"[init] pair dataset ready: {len(ds)} pairs", flush=True)

    print(
        f"[init] building dataloader (batch={args['batch_size']}, "
        f"grad_accum={args['grad_accum']}, "
        f"workers={args['workers']}, prefetch={args['prefetch']})...",
        flush=True,
    )
    print(
        f"[init] effective batch size: "
        f"{args['batch_size'] * args['grad_accum']} "
        f"(physical {args['batch_size']} x grad_accum {args['grad_accum']})",
        flush=True,
    )
    dl = tc.utils.data.DataLoader(
        ds,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["workers"],
        collate_fn=collate_pairs,
        drop_last=True,
        prefetch_factor=args["prefetch"],
    )

    print("[init] creating mixup...", flush=True)
    mixup = Mixup()

    print("[init] building model...", flush=True)
    model = SiameseEncoderBT(proj_dims=1024).to(device)

    optimizer = tc.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=10
    )
    start_epoch: int = 0
    if checkpoint:
        print("[init] restoring state from checkpoint...", flush=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[init] restored, resuming at epoch {start_epoch}", flush=True)

    loss_fn = BarlowTwinsLoss(lambd=args["lambd"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = pathlib.Path(f"train_log_{timestamp}.csv")
    print(f"logging metrics to {log_path}")

    train(
        model,
        dl,
        optimizer,
        scheduler,
        loss_fn,
        mixup=mixup,
        grad_accum=args["grad_accum"],
        epochs=epochs,
        start_epoch=start_epoch,
        log_path=log_path,
    )
