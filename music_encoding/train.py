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

from music_encoding.augmenter import Mixup, SpectrogramAugmenter
from music_encoding.model import BarlowTwinsLoss, InfoNCELoss, SiameseEncoderBT
from music_encoding.mtg import MTGJamendoBase
from music_encoding.twin_dataset import FMAPairDataset, MelSpecPairDataset, collate_pairs


def train(
    model: SiameseEncoderBT,
    dl: DataLoader,
    optimizer: Optimizer,
    scheduler: tc.optim.lr_scheduler.ReduceLROnPlateau,
    loss_fn: Callable[[tc.Tensor, tc.Tensor], tuple[tc.Tensor, tc.Tensor, tc.Tensor]],
    mixup: Mixup | None = None,
    epochs: int = 300,
    device: str = "cuda",
    start_epoch: int = 0,
    log_path: pathlib.Path | None = None,
    checkpoint_path: os.PathLike | None = None,
    # None => no autocast (plain fp32); fp16/bf16 run under autocast(device_type="cuda")
    autocast_dtype: tc.dtype | None = tc.bfloat16,
    # inter-sample repulsion (see InfoNCELoss). nce_fn None or nce_weight 0 => the
    # objective is plain Barlow Twins, i.e. unchanged from every earlier run.
    # nce_on picks WHICH representation is repelled: "z" (the projector, what BT
    # shapes) or "emb" (128-d, what the DB stores and retrieval is measured on).
    nce_fn: Callable[[tc.Tensor, tc.Tensor], tc.Tensor] | None = None,
    nce_weight: float = 0.0,
    nce_on: str = "z",
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
                ["epoch", "lr", "loss", "on_diag", "off_diag", "off_diag_norm", "nce"]
            )

    print(
        f"[train] starting: epochs={start_epoch}..{start_epoch + epochs - 1}, "
        f"batches/epoch={len(dl)}, device={device}, "
        f"checkpoints -> {checkpoint_path}",
        flush=True,
    )

    try:
        for epoch in range(start_epoch, start_epoch + epochs):
            model.train()
            total_loss = 0.0
            total_on_diag = 0.0
            total_off_diag = 0.0
            total_nce = 0.0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[train] epoch {epoch}/{start_epoch + epochs - 1} (lr={lr}) ...",
                flush=True,
            )
            steps = 0
            for batch_idx, (spec_a, spec_b) in enumerate(dl):
                # flush: stdout is redirected to a log on these runs, so without it
                # this sits in the 8 KB block buffer and a killed run is
                # indistinguishable from a hung one (which cost real debugging time)
                print(f"[train] batch {batch_idx}", flush=True)
                spec_a = spec_a.to(device)
                spec_b = spec_b.to(device)

                if mixup is not None:
                    # in-batch mixing; each side gets a different partner.
                    # The dataset already did waveform aug + mel + RRC, so Mixup
                    # now blends the resulting spectrograms.
                    spec_a = mixup(spec_a)
                    spec_b = mixup(spec_b)

                # No gradient accumulation, by design: the BT loss is a
                # batch-normalized statistic (the projector's BN uses this
                # batch's mean/std, and the cross-correlation couples all
                # samples), so it does NOT decompose into independent
                # per-sample terms. Summing micro-batch gradients would NOT
                # equal one larger batch — each DataLoader batch IS the BT
                # batch. If more samples are wanted, raise --batch-size
                # directly (BT saturates at 64-512 anyway).
                optimizer.zero_grad()

                with tc.autocast(
                    device_type="cuda",
                    enabled=autocast_dtype is not None,
                    dtype=autocast_dtype if autocast_dtype is not None else tc.float32,
                ):
                    emb_a, emb_b, z_a, z_b = model.forward_pair(spec_a, spec_b)
                    loss, on_diag, off_diag = loss_fn(z_a, z_b)
                    if nce_fn is not None and nce_weight > 0:
                        # inter-sample repulsion on top of BT; logged separately so
                        # the existing columns keep their exact meaning
                        if nce_on == "emb":
                            nce = nce_fn(emb_a, emb_b)
                        else:
                            nce = nce_fn(z_a, z_b)
                        loss = loss + nce_weight * nce

                loss.backward()
                optimizer.step()
                steps += 1
                total_loss += loss.item()
                total_on_diag += on_diag.item()
                total_off_diag += off_diag.item()
                if nce_fn is not None and nce_weight > 0:
                    total_nce += float(nce.detach())

                if steps % 50 == 0:
                    print(
                        f"[train] epoch {epoch} step {steps} (batch {batch_idx}/{len(dl)}) "
                        f"loss={loss.item():.4f}",
                        flush=True,
                    )

            avg_loss = total_loss / len(dl)
            # ReduceLROnPlateau takes the metric; the others take none. The plateau
            # variant is provably inert here (43 epochs at a constant lr, since the
            # ~-1/epoch descent keeps beating its relative threshold), which is why
            # a schedule that actually anneals is worth having for long runs.
            if isinstance(scheduler, tc.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            on_diag_val = total_on_diag / len(dl)
            off_diag_val = total_off_diag / len(dl)
            # mean-squared off-diagonal correlation, normalized by the D*(D-1) terms
            off_diag_norm = off_diag_val / (D * (D - 1))
            off_diag_mean_abs = math.sqrt(off_diag_norm)  # mean |c_ij|

            nce_val = total_nce / len(dl) if (nce_fn is not None and nce_weight > 0) else 0.0

            print(
                f"epoch: {epoch}",
                f"lr: {current_lr}",
                f"loss: {avg_loss}",
                f"on_diag: {on_diag_val}",
                f"off_diag: {off_diag_val}",
                f"off_diag_norm: {off_diag_norm}",
                f"off_diag_mean_abs: {off_diag_mean_abs:.4f}",
                f"nce: {nce_val:.4f} (w={nce_weight:g}, share={nce_weight * nce_val / avg_loss:.1%})",
                sep=", ",
                flush=True,
            )

            if csv_writer is not None and log_file is not None:
                csv_writer.writerow(
                    [
                        epoch,
                        current_lr,
                        avg_loss,
                        on_diag_val,
                        off_diag_val,
                        off_diag_norm,
                        nce_val,
                    ]
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
    parser.add_argument(
        "--cache-dir",
        default=pathlib.Path("cached_mtg"),
        type=pathlib.Path,
        help="dir for cached resampled full-track waveforms; pass 'none' to "
             "decode per-epoch instead (use when disk can't hold the cache)",
    )
    parser.add_argument(
        "--mel-root", default=None, type=pathlib.Path,
        help="dir of precomputed MTG-Jamendo log-mel .npy files, laid out as "
             "<id mod 100>/<id>.npy. When set, TRAIN ON PRECOMPUTED SPECS: no audio "
             "decode, no mel conversion, no waveform augmentation; windows are drawn "
             "straight from the spectrogram. Mutually exclusive with --mtg-data "
             "(the waveform path).",
    )
    parser.add_argument(
        "--mtg-data", default=None, type=pathlib.Path,
        help="MTG-Jamendo data dir (contains autotagging.tsv, raw.meta.tsv); "
             "used by the waveform path only (when --mel-root is unset)",
    )
    parser.add_argument(
        "--audio-root", default=None, type=pathlib.Path,
        help="dir where MTG audio unpacked (defaults to --mtg-data)",
    )
    parser.add_argument(
        "--min-window-rms", default=0.03, type=float,
        help="(waveform path) re-draw windows quieter than this RMS "
             "(near-silence); 0 disables",
    )
    parser.add_argument(
        "--min-window-db", default=-85.0, type=float,
        help="(mel path) re-draw spectrogram windows whose mean is below this dB; "
             "default -85. Pass < -90 (the power floor) to disable",
    )
    parser.add_argument(
        "--n-mels", default=64, type=int,
        help="(mel path) model-input mel bands. The .npy specs are native (96, T); "
             "the default 64 merges them down (in linear-power space) to match the "
             "64-mel waveform/audio-eval path (LogMelSpectrogram n_mels=64). Pass 96 "
             "to keep the native resolution. Ignored on the waveform path.",
    )
    parser.add_argument("--lambd", default=5e-2, type=float)
    parser.add_argument(
        "--lr-schedule", default="plateau", choices=["plateau", "cosine", "constant"],
        help="'plateau' (default, unchanged) = ReduceLROnPlateau(factor=0.1, "
             "patience=10) on avg_loss; it holds 1e-4 for ~295 epochs then fires "
             "once. 'cosine' = CosineAnnealingLR down to --lr-min (measured "
             "net-negative while the loss is still descending). 'constant' = fixed "
             "1e-4 forever, implemented as LambdaLR(lr_lambda=1) so it keeps the "
             "same step()/state_dict() interface. Use 'constant' for long compute "
             "pushes: the plateau scheduler re-fires within ~20 epochs of a resume "
             "even after --reset-lr, and 1e-5 makes the remaining epochs crawl.",
    )
    parser.add_argument(
        "--lr-min", default=1e-5, type=float,
        help="floor for --lr-schedule cosine (default 1e-5)",
    )
    parser.add_argument(
        "--reset-lr", action="store_true",
        help="after resuming, restore the optimizer's LR to 1e-4 and rebuild the "
             "scheduler, discarding the checkpoint's patience state. Use when "
             "continuing a run whose LR the plateau scheduler already dropped: it "
             "fires on a LOSS plateau, but a loss plateau this objective produces "
             "does not mean the eval metric has saturated (retrieval kept improving "
             "0.525 -> 0.579 while the loss looked flat), so annealing early just "
             "makes the remaining steps shorter. Measured: the scheduler holds 1e-4 "
             "for ~295 epochs and then fires once.",
    )
    parser.add_argument(
        "--nce-weight", default=0.0, type=float,
        help="weight on the InfoNCE inter-sample-repulsion term added on top of "
             "Barlow Twins (0 = off, i.e. plain BT). BT has no term pushing "
             "DIFFERENT tracks apart, so unrelated windows sit at cosine ~0.83 and "
             "retrieval top-1 stalls ~0.44. NOTE the scales differ hugely: BT "
             "totals ~640 while InfoNCE is ~ln(2B-1) ~ 6.6, so a useful weight is "
             "O(10), not a small fraction. The epoch log prints the term's share of "
             "the total loss so the balance is visible.",
    )
    parser.add_argument(
        "--nce-temp", default=0.1, type=float,
        help="InfoNCE temperature (default 0.1, the SimCLR value). Lower = harder "
             "negatives.",
    )
    parser.add_argument(
        "--nce-on", default="z", choices=["z", "emb"],
        help="which representation the repulsion acts on. 'z' = the projector "
             "(what BT shapes, SimCLR convention); 'emb' = the 128-d embedding that "
             "build_chroma stores and test_retrieval measures. Measured: NCE on 'z' "
             "at weight 30 left whitened retrieval unchanged (0.362 -> 0.368) while "
             "raising collapse 50 -> 167 per 1000, so 'emb' is worth trying — the "
             "retrieved space is only shaped indirectly through z.",
    )
    parser.add_argument(
        "--autocast", default="bf16", choices=["bf16", "fp16", "fp32"],
        help="forward dtype under autocast. 'fp16' = native half precision "
             "(good on CUDA Turing+ and ROCm RDNA — RX 6xxx/7xxx has NATIVE "
             "fp16, ~2x fp32 speed, half the activation memory); 'bf16' = "
             "bfloat16 (good on CUDA Ampere+ / ROCm CDNA, but consumer RDNA "
             "emulates it ~2.4x SLOWER than fp32); 'fp32' = no autocast. "
             "The BT loss upcasts to fp32 internally regardless.",
    )

    args = vars(parser.parse_args(sys.argv[1:]))
    resume: pathlib.Path | None = args["resume"]
    epochs: int = args["epochs"]

    # two mutually exclusive data sources: precomputed mel specs, or MTG audio
    if (args["mel_root"] is None) == (args["mtg_data"] is None):
        print(
            "[init] ERROR: set exactly one of --mel-root (precomputed log-mel "
            "specs, no conversion) or --mtg-data (waveform audio path)",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(2)

    print(f"[init] device={device} resume={resume} epochs={epochs}", flush=True)

    print("[init] loading checkpoint...", flush=True)
    checkpoint = (
        tc.load(resume, map_location=device)
        if resume is not None and resume.is_file()
        else None
    )
    print(f"[init] checkpoint loaded: {resume if checkpoint else 'none'}", flush=True)

    if args["mel_root"] is not None:
        print("[init] MEL path: reading precomputed log-mel .npy under "
              f"{args['mel_root']} (no audio decode, no mel conversion)", flush=True)
        print("[init] creating spectrogram augmenter "
              "(content distortions + RRC)...", flush=True)
        spectrogram_augmenter = SpectrogramAugmenter()
        print("[init] building pair dataset...", flush=True)
        ds = MelSpecPairDataset(
            args["mel_root"],
            spectrogram_augmenter=spectrogram_augmenter,
            min_window_db=args["min_window_db"],
            n_mels=args["n_mels"],
        )
        print(f"[init] pair dataset ready: {len(ds)} tracks "
              f"(waveform augmenter & cache: N/A in mel mode)", flush=True)
    else:
        print("[init] loading MTG-Jamendo base dataset...", flush=True)
        base_ds = MTGJamendoBase(args["mtg_data"], audio_root=args["audio_root"])
        print(f"[init] base dataset loaded: {len(base_ds)} tracks", flush=True)

        print("[init] creating spectrogram augmenter "
              "(content distortions + RRC)...", flush=True)
        spectrogram_augmenter = SpectrogramAugmenter()

        cache_dir = None if str(args["cache_dir"]) == "none" else args["cache_dir"]
        cache_note = "DISABLED (decode per-epoch)" if cache_dir is None else cache_dir
        print(f"[init] building pair dataset (cache: {cache_note})...", flush=True)
        # NB: no waveform AudioAugmenter here — all distortions are applied by
        # SpectrogramAugmenter on the log-mel (see augmenter.py), so both the
        # waveform and the precomputed-mel data paths distort identically.
        ds = FMAPairDataset(
            base_ds,
            cache_dir=cache_dir,
            spectrogram_augmenter=spectrogram_augmenter,
            min_window_rms=args["min_window_rms"],
        )
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
        # DataLoader rejects prefetch_factor when num_workers=0, and 0 workers is the
        # escape hatch when worker spawn fails (Python 3.14 defaults multiprocessing
        # to forkserver, whose dataset pickling has been truncating on this box)
        prefetch_factor=args["prefetch"] if args["workers"] > 0 else None,
    )

    # Mixup is intentionally OFF in the default pipeline: with the current
    # positives (two different windows of one track, each independently
    # augmented) it measurably worsens the already-hard time-stability
    # objective (ablation: diff-windows+aug on_diag 802 with mixup vs 518
    # without). train() still accepts mixup=None as an experiment hook.
    print("[init] building model...", flush=True)
    model = SiameseEncoderBT(proj_dims=1024).to(device)

    optimizer = tc.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    if args["lr_schedule"] == "cosine":
        scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=args["lr_min"]
        )
        print(
            f"[init] lr schedule: cosine 1e-4 -> {args['lr_min']:g} over {epochs} epochs",
            flush=True,
        )
    elif args["lr_schedule"] == "constant":
        # a no-op scheduler: LambdaLR keeps the same step()/state_dict() interface,
        # so nothing else in train() has to special-case it
        scheduler = tc.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
        print("[init] lr schedule: constant 1e-4 (never annealed)", flush=True)
    else:
        scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10
        )
    start_epoch: int = 0
    if checkpoint:
        print("[init] restoring state from checkpoint...", flush=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # skip the old scheduler state when --reset-lr discards it anyway: the
        # checkpoint may have been written by a different scheduler class (e.g.
        # resuming a plateau run as --lr-schedule constant), and the state dicts
        # are not interchangeable
        if not args["reset_lr"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"[init] restored, resuming at epoch {start_epoch}", flush=True)
        if args["reset_lr"]:
            base_lr = 1e-4
            for group in optimizer.param_groups:
                group["lr"] = base_lr
            if args["lr_schedule"] == "cosine":
                scheduler = tc.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=args["lr_min"]
                )
            elif args["lr_schedule"] == "constant":
                scheduler = tc.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=lambda _: 1.0
                )
            else:
                scheduler = tc.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.1, patience=10
                )
            print(
                f"[init] --reset-lr: LR restored to {base_lr:g} with a fresh scheduler "
                f"(the checkpoint's reduced LR / patience state is discarded)",
                flush=True,
            )

    loss_fn = BarlowTwinsLoss(lambd=args["lambd"])
    nce_fn = InfoNCELoss(temperature=args["nce_temp"]) if args["nce_weight"] > 0 else None
    if nce_fn is not None:
        print(
            f"[init] inter-sample repulsion ON: InfoNCE weight={args['nce_weight']}, "
            f"temperature={args['nce_temp']}",
            flush=True,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = pathlib.Path(f"train_log_{timestamp}.csv")
    print(f"logging metrics to {log_path}")

    train(
        model,
        dl,
        optimizer,
        scheduler,
        loss_fn,
        mixup=None,  # Mixup dropped from the default pipeline (see above)
        epochs=epochs,
        start_epoch=start_epoch,
        log_path=log_path,
        nce_fn=nce_fn,
        nce_weight=args["nce_weight"],
        nce_on=args["nce_on"],
        autocast_dtype={
            "fp16": tc.float16,
            "bf16": tc.bfloat16,
            "fp32": None,
        }[args["autocast"]],
    )
