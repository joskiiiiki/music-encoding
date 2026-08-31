#!/usr/bin/env bash
# setup_and_train.sh — RunPod 3090 one-shot: download MTG audio, fix the
# autotagging TSV stub, train the Barlow Twins encoder, then stop the pod so
# billing ends. The steps run sequentially: download && tsv-fix && train && stop.
#
# Launch it under nohup so it survives SSH disconnect:
#   nohup bash setup_and_train.sh > /workspace/train.log 2>&1 &
# (If the image was built with `COPY . /app`, it's at /app/setup_and_train.sh.)
#
# Safe to re-run after a pod restart: the TSV fix is idempotent and training
# resumes from a checkpoint when RESUME is set. Note the download script skips
# tars that already exist, so an interrupted download resumes at tar granularity.
#
# Assumed layout (see CLAUDE.md / music_encoding/mtg.py):
#   /workspace/mtg/                clone of github.com/MTG/mtg-jamendo-dataset
#       scripts/download/download.py   the official downloader (needs gdown)
#       data/autotagging.tsv       stub (31 bytes) -> replaced below
#       data/raw_30s_cleantags_50artists.tsv   the real tag file (55,609 tracks)
#       data/raw.meta.tsv
#   /workspace/audio/audio-low/    unpacked audio-low download (this is AUDIO_ROOT)
#   /app/                          this repo (Dockerfile WORKDIR + COPY . /app)
#
# Env overrides (set as pod environment variables):
#   RUNPOD_API_KEY   if set, the pod is stopped (runpodctl stop) after training
#                    exits 0. Without it the script just finishes and leaves the
#                    pod running. Grab the key at runpod.io -> Settings -> API Keys.
#   MTG_DATA   MTG_DIR   DL_DIR   EPOCHS   BATCH_SIZE   GRAD_ACCUM
#   WORKERS (default: nproc)   PREFETCH   RESUME (checkpoint to resume from)

set -euo pipefail

MTG_DATA="${MTG_DATA:-/workspace/mtg/data}"
MTG_DIR="${MTG_DIR:-/workspace/mtg}"
DL_DIR="${DL_DIR:-/workspace/audio}"           # download target dir
AUDIO_ROOT="${AUDIO_ROOT:-$DL_DIR/audio-low}"  # tars unpack to $DL_DIR/<type>/
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-768}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
WORKERS="${WORKERS:-$(nproc)}"
PREFETCH="${PREFETCH:-2}"

# train.py writes checkpoints relative to cwd -> run from /app (the repo root).
cd /app

# ---------------------------------------------------------------------------
# 1. Download the audio (audio-low, 156 GB unpacked), then unpack + delete tars.
#    The official script skips tars that already exist, so an interrupted run
#    resumes at tar granularity.
# ---------------------------------------------------------------------------
echo "[setup] $(date +%H:%M:%S) starting audio download -> $DL_DIR"
python3 "$MTG_DIR/scripts/download/download.py" \
    --dataset raw_30s --type audio-low "$DL_DIR" --unpack --remove
echo "[setup] $(date +%H:%M:%S) audio download finished"

# ---------------------------------------------------------------------------
# 2. The checked-in data/autotagging.tsv is a 31-byte stub. Point it at the real
#    tag file. Idempotent: overwrite with a backup of the stub kept aside.
# ---------------------------------------------------------------------------
if [ -f "$MTG_DATA/raw_30s_cleantags_50artists.tsv" ]; then
    cp -f "$MTG_DATA/autotagging.tsv" "$MTG_DATA/autotagging.tsv.stub" 2>/dev/null || true
    cp -f "$MTG_DATA/raw_30s_cleantags_50artists.tsv" "$MTG_DATA/autotagging.tsv"
    echo "[setup] autotagging.tsv <- raw_30s_cleantags_50artists.tsv"
else
    echo "[setup] WARNING: $MTG_DATA/raw_30s_cleantags_50artists.tsv not found — " \
        "leaving autotagging.tsv as-is (train will likely fail on the stub)"
fi

# ---------------------------------------------------------------------------
# 3. Quick sanity check: audio_root should hold the unpacked mp3s.
# ---------------------------------------------------------------------------
N_AUDIO="$(find "$AUDIO_ROOT" -type f -name '*.mp3' 2>/dev/null | wc -l)"
echo "[setup] found $N_AUDIO mp3s under $AUDIO_ROOT (expect ~55.6k)"
if [ "$N_AUDIO" -lt 1000 ]; then
    echo "[setup] ERROR: audio looks wrong — refusing to train. Fix AUDIO_ROOT."
    exit 1
fi

# ---------------------------------------------------------------------------
# 4. Train. set -e means a non-zero exit from train.py aborts the script before
#    the pod-stop step, leaving the pod up so you can debug.
# ---------------------------------------------------------------------------
RESUME_ARGS=()
if [ -n "${RESUME:-}" ]; then
    RESUME_ARGS=("-r" "$RESUME")
fi

echo "[setup] $(date +%H:%M:%S) starting training: ${EPOCHS} epochs, " \
    "batch ${BATCH_SIZE}, grad_accum ${GRAD_ACCUM} (effective $((BATCH_SIZE * GRAD_ACCUM))), " \
    "${WORKERS} workers, prefetch ${PREFETCH}"
python -m music_encoding.train \
    --mtg-data "$MTG_DATA" \
    --audio-root "$AUDIO_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --workers "$WORKERS" \
    --prefetch "$PREFETCH" \
    -e "$EPOCHS" \
    "${RESUME_ARGS[@]}"

# ---------------------------------------------------------------------------
# 5. Training exited 0 -> stop the pod to end billing (only if a key is set).
# ---------------------------------------------------------------------------
if [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo "[setup] training done — stopping pod ${RUNPOD_POD_ID:-unknown}"
    RPCTL="$(command -v runpodctl || true)"
    if [ -z "$RPCTL" ]; then
        echo "[setup] installing runpodctl..."
        curl -sSfL https://raw.githubusercontent.com/runpod/runpodctl/main/install.sh | sh
        RPCTL="$HOME/.local/bin/runpodctl"
    fi
    export RUNPOD_API_KEY
    "$RPCTL" stop pod "${RUNPOD_POD_ID}"
else
    echo "[setup] training done — RUNPOD_API_KEY not set, leaving the pod running"
    echo "[setup] (set RUNPOD_API_KEY as a pod env var to auto-stop after training)"
fi
