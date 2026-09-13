#!/usr/bin/env python3
"""Embed arbitrary audio into the corpus's embedding space.

Run in the **default** dev shell -- that is the one with torch::

    nix develop --command python webapp/embedder/service.py

Why this is a separate process
------------------------------
The API runs in the light ``.#web`` shell, which deliberately has no torch (it only ever
read exported artifacts). Turning a Deezer/iTunes hit into something comparable with the
corpus needs the checkpoint and MTG's mel front end, so that work lives here and the API
calls it over HTTP. Keeping it out of the API process also keeps ~600 MB of torch out of
the thing that serves every browse and audio request.

Nothing here is reimplemented: ``front_end`` and ``embed_preview`` are imported from
``music_encoding.compare_songs``, which already reproduces MTG's front end (96 mel @
24 kHz, slaney) and pools five 5-second windows exactly the way ``build_chroma`` did.

Caveat worth stating plainly: a preview is a *different recording* of the song, so a
query embedding carries more error than a corpus one. ``compare_songs`` measured ~0.987
cosine between the audio and .npy paths *for the same recording*; against a different
recording the mismatch is larger and the neighbouring tracks are correspondingly
noisier.
The API surfaces this so the UI can say so.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CHECKPOINT = os.environ.get(
    "EMBEDDER_CHECKPOINT",
    "/home/johannes/mel_runs/cont1000/checkpoints/checkpoint_e1000.pt",
)
# Local paths are only accepted under here: this endpoint reads files, and it should not
# become a way to read anything on the machine.
LOCAL_ROOT = pathlib.Path(
    os.environ.get("EMBEDDER_LOCAL_ROOT", pathlib.Path.home() / "mtg")
)
WINDOWS = 5

MODEL = None
FRONT_END = None
EMBED_PREVIEW = None
FRONT_SR = None


def load(checkpoint: str) -> None:
    """Load the checkpoint and the front end once, at startup."""
    global MODEL, FRONT_END, EMBED_PREVIEW, FRONT_SR
    import torch

    from music_encoding.compare_songs import FRONT_SR as SR
    from music_encoding.compare_songs import embed_preview as ep
    from music_encoding.compare_songs import front_end as fe
    from music_encoding.model import SiameseEncoderBT

    started = time.time()
    ck = torch.load(checkpoint, map_location="cuda", weights_only=False)
    model = SiameseEncoderBT(
        proj_dims=ck["model_state_dict"]["projector.6.weight"].shape[0]
    ).to("cuda")
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    MODEL, FRONT_END, EMBED_PREVIEW, FRONT_SR = model, fe, ep, SR
    print(
        f"embedder ready: {checkpoint} (epoch {ck.get('epoch', '?')}) "
        f"in {time.time() - started:.1f}s",
        flush=True,
    )


def embed(audio: pathlib.Path) -> tuple[list[float], float, list[int]]:
    """Audio file -> (128-d raw embedding, seconds of audio, mel shape)."""
    import torchaudio
    from torchcodec.decoders import AudioDecoder

    samples = AudioDecoder(str(audio)).get_all_samples()
    wav = samples.data
    if wav.ndim > 1:
        wav = wav.mean(dim=0)  # mono, matching the corpus
    if samples.sample_rate != FRONT_SR:
        wav = torchaudio.functional.resample(wav, samples.sample_rate, FRONT_SR)
    spec = FRONT_END(wav)  # the MTG front end, 96 mel -> 64
    vector = EMBED_PREVIEW(MODEL, spec, WINDOWS)  # mean of 5 x 5s windows
    return vector.tolist(), wav.shape[-1] / FRONT_SR, list(spec.shape[1:])


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.startswith("/health"):
            self._send(
                200,
                {"ok": MODEL is not None, "dim": 128, "checkpoint": CHECKPOINT},
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if not self.path.startswith("/embed"):
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as exc:
            self._send(400, {"error": f"bad request body: {exc}"})
            return

        started = time.time()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                target = pathlib.Path(tmp) / "audio"
                if url := request.get("url"):
                    from music_encoding.compare_songs import _download

                    _download(url, target)
                    source = url
                elif path := request.get("path"):
                    candidate = pathlib.Path(path).resolve()
                    # Reads local files, so refuse anything outside the audio root.
                    if not candidate.is_relative_to(LOCAL_ROOT.resolve()):
                        self._send(403, {"error": f"path outside {LOCAL_ROOT}"})
                        return
                    if not candidate.is_file():
                        self._send(404, {"error": "no such file"})
                        return
                    target = candidate
                    source = str(candidate)
                else:
                    self._send(400, {"error": "need 'url' or 'path'"})
                    return

                vector, seconds, mel_shape = embed(target)
        except Exception as exc:  # noqa: BLE001 - report any failure to the caller
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        self._send(
            200,
            {
                "vector": vector,
                "dim": len(vector),
                "audio_seconds": round(seconds, 2),
                "mel_shape": mel_shape,
                "source": source,
                "elapsed_ms": round((time.time() - started) * 1000),
            },
        )

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(f"[embedder] {fmt % args}\n")


CHECKPOINT = DEFAULT_CHECKPOINT


def main() -> int:
    global CHECKPOINT
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--port", type=int, default=int(os.environ.get("EMBEDDER_PORT", 8100))
    )
    args = ap.parse_args()
    CHECKPOINT = args.checkpoint

    if not pathlib.Path(CHECKPOINT).exists():
        raise SystemExit(f"checkpoint not found: {CHECKPOINT}")
    load(CHECKPOINT)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
