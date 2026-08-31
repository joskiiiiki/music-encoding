# RunPod training image — RTX 3090 (sm_86, CUDA 13.0).
#
# Bases on the official PyTorch image for torch 2.13.0 + CUDA 13.0, which already
# ships the matching cu130 torch/torchvision/torchaudio (Ampere sm_86 is still
# supported by CUDA 13.0 — it dropped pre-Ampere Maxwell/Pascal/Volta, which we
# don't need). We re-pin them explicitly and add torchcodec from the same cu130
# index so every CUDA component comes from one source. NOTE: torch 2.6 on the
# cu124 index was tried first — it can't work here because torchcodec < 0.3 has
# no AudioDecoder, and AudioDecoder needs torch >= 2.7.
#
# Build + push (from repo root):
#   docker build -t <your-registry>/music-encoding:3090 .
#   docker push <your-registry>/music-encoding:3090
#
# Run on RunPod (3090): mount a volume with the MTG-Jamendo data + audio, then:
#   python -m music_encoding.train --mtg-data /workspace/mtg/data \
#       --audio-root /workspace/mtg/audio --cache-dir /workspace/cache -e 300
#   python -m music_encoding.build_chroma checkpoints/checkpoint_e99.pt \
#       --mtg-data /workspace/mtg/data --audio-root /workspace/mtg/audio
#
# NOTE: bfloat16 autocast works on the 3090. `torch.compile` is left disabled
# (it was dropped for ROCm; re-enable on CUDA if you want it).

FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

# audio decode (torchcodec / torchaudio) needs ffmpeg + libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# This base is a Debian system Python (no conda), so pip hits PEP 668's
# externally-managed guard -> --break-system-packages (fine inside a container).
# torch/torchvision/torchaudio already ship in the base as cu130 builds; pin them
# explicitly so the image doesn't drift if the base changes, and add torchcodec
# (not bundled) from the same cu130 index — its AudioDecoder is what decodes the
# MTG mp3s
RUN pip install --break-system-packages --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.13.0 torchvision==0.28.0 torchaudio==2.11.0 torchcodec==0.16.0

# rest of the Python deps (chroma is the vec DB for the evals, pyvis for the
# interactive network HTML, gdown for the MTG download script)
RUN pip install --break-system-packages --no-cache-dir \
        numpy datasets transformers matplotlib pillow \
        chromadb pyvis gdown

WORKDIR /app
COPY . /app

# keep the container alive so you can exec in / set the real command on RunPod
CMD ["sleep", "infinity"]
