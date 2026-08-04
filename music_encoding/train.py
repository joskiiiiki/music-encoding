from music_encoding.augmenter import AudioAugmenter
import os
from typing import cast
from importlib.resources import path
import pathlib
import sys
import csv
import argparse
from datetime import datetime
from collections.abc import Callable
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import datasets
from torch import nn
import torch as tc
from music_encoding.model import BarlowTwinsLoss, SiameseEncoderBT
from music_encoding.twin_dataset import FMAPairDataset, collate_pairs


def train(
    model: SiameseEncoderBT,
    dl: DataLoader,
    optimizer: Optimizer,
    scheduler: tc.optim.lr_scheduler.ReduceLROnPlateau,
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tuple[tc.Tensor, tc.Tensor, tc.Tensor]],
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

    # open once, append per epoch — avoids reopening/rewriting the file every iteration
    log_file = None
    csv_writer = None
    if log_path is not None:
        is_new_file = not log_path.exists()
        log_file = open(log_path, "a", newline="")
        csv_writer = csv.writer(log_file)
        if is_new_file:
            csv_writer.writerow(["epoch", "lr", "loss", "on_diag", "off_diag"])

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            total_loss = 0.0
            for wav_a, wav_b in dl:
                wav_a = wav_a.to(device)
                wav_b = wav_b.to(device)
                optimizer.zero_grad()
                with tc.autocast(device_type="cuda", dtype=tc.bfloat16):
                    emb_a, emb_b, z_a, z_b = model.forward_pair(wav_a, wav_b)
                    loss, on_diag, off_diag = loss_fn(z_a, z_b)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dl)
            scheduler.step(avg_loss)

            current_lr = optimizer.param_groups[0]["lr"]
            on_diag_val = on_diag.item()
            off_diag_val = off_diag.item()

            print(
                f"epoch: {epoch}",
                f"lr: {current_lr}",
                f"loss: {avg_loss}",
                f"on_diag: {on_diag_val}",
                f"off_diag: {off_diag_val}",
                sep=", ",
            )

            if csv_writer is not None and log_file is not None:
                csv_writer.writerow(
                    [epoch, current_lr, avg_loss, on_diag_val, off_diag_val]
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
    parser.add_argument("--prefetch", default=2, type=int)

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

    print("[init] loading base dataset...", flush=True)
    base_ds = datasets.load_dataset("benjamin-paine/free-music-archive-small")["train"]
    print(f"[init] base dataset loaded: {len(base_ds)} tracks", flush=True)

    print("[init] creating augmenter...", flush=True)
    augmenter = AudioAugmenter()

    print("[init] building pair dataset (caching resampled tracks)...", flush=True)
    ds = FMAPairDataset(base_ds, cache_dir="./cached", augmenter=augmenter)
    print(f"[init] pair dataset ready: {len(ds)} pairs", flush=True)

    print(
        f"[init] building dataloader (batch={args['batch_size']}, "
        f"workers={args['workers']}, prefetch={args['prefetch']})...",
        flush=True,
    )
    dl = tc.utils.data.DataLoader(
        ds,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["workers"],
        collate_fn=collate_pairs,
        drop_last=True,
        prefetch_factor=args["prefetch"]
    )

    print("[init] building model...", flush=True)
    model = SiameseEncoderBT(proj_dims=2048).to(device)
    print("[init] compiling model forward pass (this can take a while)...", flush=True)
    model.forward = tc.compile(model.forward)
    print("[init] model compiled", flush=True)

    optimizer = tc.optim.Adam(model.parameters(), lr=1e-4)
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

    print("[init] compiling loss function (this can take a while)...", flush=True)
    loss_fn = tc.compile(BarlowTwinsLoss(lambd=4.9e-4))
    print("[init] loss compiled", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = pathlib.Path(f"train_log_{timestamp}.csv")
    print(f"logging metrics to {log_path}")

    train(
        model,
        dl,
        optimizer,
        scheduler,
        loss_fn,
        epochs=epochs,
        start_epoch=start_epoch,
        log_path=log_path,
    )
