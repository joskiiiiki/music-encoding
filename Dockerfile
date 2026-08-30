# RunPod training image — RTX 3090 (CUDA 12.x, sm_86).
#
# Bases on the official PyTorch CUDA image (python + CUDA + torch preinstalled),
# then bumps torch/torchaudio/torchcodec to the same latest matching set used in
# the local nix env (torch 2.12 / torchaudio 2.11 / torchcodec 0.14). On Linux,
# `pip install torch*` pulls CUDA wheels from PyPI, so no index URL is needed.
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

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# audio decode (torchcodec / torchaudio) needs ffmpeg + libsndfile
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# keep torch/torchvision/torchaudio/torchcodec on the same latest release line
RUN pip install --no-cache-dir --upgrade torch torchvision torchaudio torchcodec

# rest of the Python deps (chroma is the vec DB for the evals, pyvis for the
# interactive network HTML, gdown for the MTG download script)
RUN pip install --no-cache-dir \
        numpy datasets transformers matplotlib pillow \
        chromadb pyvis gdown

WORKDIR /app
COPY . /app

# keep the container alive so you can exec in / set the real command on RunPod
CMD ["sleep", "infinity"]
