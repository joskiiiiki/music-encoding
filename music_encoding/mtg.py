"""Official MTG-Jamendo dataset loader.

Wraps the official MTG-Jamendo dataset (github.com/MTG/mtg-jamendo-dataset):
  - data/autotagging.tsv  TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS
  - data/raw.meta.tsv     TRACK_ID ARTIST_ID ALBUM_ID TRACK_NAME ARTIST_NAME
                          ALBUM_NAME RELEASEDATE URL
with the audio unpacked under an `audio_root` — download it with:

  python3 scripts/download/download.py --dataset raw_30s --type audio-low \\
      <dir> --unpack --remove        # from the mtg-jamendo-dataset repo

Each track carries exactly one `category---tag` in autotagging.tsv (mostly
`genre---x`), so genre/instrument/mood_theme are populated from that single tag.

Exposes an HF-Dataset-like interface so the rest of the pipeline
(FMAPairDataset, build_chroma, test_similar_pairs) works unchanged:
  len(ds), ds[idx]["audio"].get_all_samples(), ds.row(idx) for metadata.
"""

import pathlib

from torchcodec.decoders import AudioDecoder

AUTOTAGGING_TSV = "autotagging.tsv"
META_TSV = "raw.meta.tsv"


def _read_autotagging(tsv: pathlib.Path) -> dict:
    rows: dict[str, dict] = {}
    with open(tsv, newline="") as f:
        f.readline()  # header
        for line in f:
            # TAGS holds multiple `category---tag` entries separated by
            # whitespace (incl. tabs), so keep it whole after the 5th column
            tid, aid, alid, relpath, dur, tags = line.rstrip("\n").split("\t", 5)
            cats: dict[str, list[str]] = {}
            for entry in tags.split():
                cat, tag = entry.split("---", 1)
                cats.setdefault(cat, []).append(tag)
            rows[tid] = {
                "track_id": tid,
                "artist_id": aid,
                "album_id": alid,
                "relpath": relpath,
                "duration": float(dur),
                "tags": cats,
            }
    return rows


def _read_meta(tsv: pathlib.Path) -> dict:
    meta: dict[str, dict] = {}
    with open(tsv, newline="") as f:
        f.readline()  # header
        for line in f:
            # URL is last; names/URL may contain tabs, so keep the tail whole
            tid, aid, alid, tname, aname, alname, released, url = line.rstrip("\n").split("\t", 7)
            meta[tid] = {
                "title": tname,
                "artist": aname,
                "album": alname,
                "released": released,
            }
    return meta


class MTGJamendoBase:
    """Dataset-like wrapper over the official MTG-Jamendo TSVs + downloaded audio."""

    def __init__(self, data_dir, audio_root=None) -> None:
        self.data_dir = pathlib.Path(data_dir)
        self.audio_root = pathlib.Path(audio_root) if audio_root else self.data_dir
        autotag = _read_autotagging(self.data_dir / AUTOTAGGING_TSV)
        meta = _read_meta(self.data_dir / META_TSV)
        # join in autotagging.tsv order; tracks without a meta row still included
        self._tracks = [{**a, **meta.get(tid, {})} for tid, a in autotag.items()]

    def __len__(self) -> int:
        return len(self._tracks)

    def row(self, idx: int) -> dict:
        """Metadata for track idx (no audio decode)."""
        t = self._tracks[idx]
        tags = t["tags"]
        genre = tags["genre"][0] if tags.get("genre") else None
        instrument = tags["instrument"][0] if tags.get("instrument") else None
        mood = tags["mood/theme"][0] if tags.get("mood/theme") else None
        return {
            "idx": idx,
            "track_id": t["track_id"],
            "artist_id": t["artist_id"],
            "album_id": t["album_id"],
            "title": t.get("title", ""),
            "artist": t.get("artist", ""),
            "album": t.get("album", ""),
            "released": t.get("released", None),
            "genre": genre,
            "instrument": instrument,
            "mood_theme": mood,
            "duration": t["duration"],
            "relpath": t["relpath"],
        }

    def __getitem__(self, idx: int) -> dict:
        return {**self.row(idx), "audio": AudioDecoder(str(self.audio_root / self._tracks[idx]["relpath"]))}
