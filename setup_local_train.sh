#!/usr/bin/env bash
# setup_local_train.sh — launch LOCAL training on the precomputed MTG-Jamendo
# mel-spectrogram download.
#
# The dataset (~/mtg_jamendo, 32,783 tracks) is already on disk as full-track
# (96, T) log-mel .npy files, so there is nothing to download, unpack, or
# convert: this script just launches training in mel mode (--mel-root), where
# FMAPairDataset's waveform->mel chain is skipped and windows are drawn straight
# from the .npy spectrograms.
#
# Launch under nohup so it survives SSH/disconnect:
#   nohup bash setup_local_train.sh > /home/johannes/music-encoding/train.log 2>&1 &
#
# Safe to re-run: training resumes from a checkpoint when RESUME is set.
#
# Env overrides:
#   REPO     (default /home/johannes/music-encoding)  repo with the flake + package
#   MEL_ROOT (default /home/johannes/mtg_jamendo)     dir of <id%100:02d>/<id>.npy
#   EPOCHS BATCH_SIZE WORKERS PREFETCH MIN_WINDOW_DB RESUME
#   AUTOCAST (fp16|bf16|fp32; default fp16 — consumer RDNA has NATIVE fp16,
#            ~2x fp32 speed and half the activation memory; bf16 is emulated
#            ~2.4x slower here. CUDA/Ampere+ boxes can set AUTOCAST=bf16.)

set -euo pipefail

REPO="${REPO:-/home/johannes/music-encoding}"
MEL_ROOT="${MEL_ROOT:-/home/johannes/mtg_jamendo}"
EPOCHS="${EPOCHS:-300}"
# Each DataLoader batch IS the BT batch (loss is batch-normalized; grad accum is
# gone). AUTOCAST defaults to fp16 and BATCH_SIZE to 256: consumer RDNA
# (RX 6xxx/7xxx) has NATIVE fp16, ~2x fp32 speed with half the activation memory
# (bf16 is emulated ~2.4x SLOWER here). Measured on this 8 GB RX 6650 XT at
# 96x234: fp16 B=256 -> 5.9G peak, ~311 ms/step; B=320 -> 7.3G; B=384 OOM.
# 256 is the comfortable pick and sits in the BT sweet spot.
# Override for a bigger GPU: BATCH_SIZE=512 AUTOCAST=bf16.
BATCH_SIZE="${BATCH_SIZE:-256}"
WORKERS="${WORKERS:-$(nproc)}"
PREFETCH="${PREFETCH:-2}"
MIN_WINDOW_DB="${MIN_WINDOW_DB:-}"
AUTOCAST="${AUTOCAST:-fp16}"

cd "$REPO"

echo "[setup] mel dataset: $MEL_ROOT ($(find "$MEL_ROOT" -name '*.npy' | wc -l) tracks)"

RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
    RESUME_ARGS=("-r" "$RESUME")
fi

echo "[setup] $(date +%H:%M:%S) training: ${EPOCHS} epochs, batch ${BATCH_SIZE}, " \
    "${WORKERS} workers, prefetch ${PREFETCH}"
nix develop . --command python -m music_encoding.train \
    --mel-root "$MEL_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --workers "$WORKERS" \
    --prefetch "$PREFETCH" \
    -e "$EPOCHS" \
    --autocast "$AUTOCAST" \
    ${MIN_WINDOW_DB:+--min-window-db "$MIN_WINDOW_DB"} \
    "${RESUME_ARGS[@]}"
