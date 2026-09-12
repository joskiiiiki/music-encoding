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

Track identity has two forms, and `row_by_num` is the bridge between them:

  TRACK_ID   `track_0000214`  — how the TSVs name a track
  track num  `214`            — the number in TRACK_ID and in the `PATH` column
                                (`<num % 100:02d>/<num>.mp3`)

The precomputed melspec `.npy` files are named `<num>.npy` under
`<num % 100:02d>/`, so their stems are **track numbers, not TRACK_IDs** — a
melspec corpus is looked up by number, never by position.
"""

import pathlib

from torchcodec.decoders import AudioDecoder

AUTOTAGGING_TSV = "autotagging.tsv"
META_TSV = "raw.meta.tsv"
# `autotagging.tsv` is often a curated SUBSET of the release (here it is a symlink
# to raw_30s_cleantags_50artists.tsv, 55.6k rows), while a melspec corpus holds the
# full download (32,783 tracks, 65 of which the subset omits). The full tag table is
# read as a fallback for exactly those omitted ids, so metadata stays complete
# instead of blank at the tail of the corpus. Absent file -> no fallback.
FULL_AUTOTAGGING_TSV = "raw_30s.tsv"


def track_num(track_id: str) -> int:
    """`track_0000214` -> `214`, the number shared with the melspec filenames."""
    return int(track_id.rsplit("_", 1)[1])


def _read_autotagging(tsv: pathlib.Path, skip: set[str] | None = None) -> dict:
    rows: dict[str, dict] = {}
    with open(tsv, newline="") as f:
        f.readline()  # header
        for line in f:
            # TAGS holds multiple `category---tag` entries separated by
            # whitespace (incl. tabs), so keep it whole after the 5th column
            tid, aid, alid, relpath, dur, tags = line.rstrip("\n").split("\t", 5)
            if skip is not None and tid in skip:
                continue  # the caller already has this row from another table
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
        self._meta_by_id = meta  # kept so the fallback below can join it too
        # join in autotagging.tsv order; tracks without a meta row still included
        self._tracks = [{**a, **meta.get(tid, {})} for tid, a in autotag.items()]
        # numeric ids in TSV order — lets a caller that has *positions* (the audio
        # path) ask for the same metadata the mel path asks for by number
        self.track_nums: list[int] = [track_num(t["track_id"]) for t in self._tracks]
        self._num_to_idx = {num: i for i, num in enumerate(self.track_nums)}
        self._fallback = self._read_fallback()

    def _read_fallback(self) -> dict[int, dict]:
        """Rows from the full tag table for numbers the primary table omits.

        Filtered while reading (via `skip`), so the extra table costs one pass and
        a few hundred dicts rather than another 56k-row copy.
        """
        path = self.data_dir / FULL_AUTOTAGGING_TSV
        if not path.exists():
            return {}
        known = {t["track_id"] for t in self._tracks}
        return {
            track_num(tid): {**row, **self._meta_by_id.get(tid, {})}
            for tid, row in _read_autotagging(path, skip=known).items()
        }

    def __len__(self) -> int:
        return len(self._tracks)

    @staticmethod
    def _fields(t: dict) -> dict:
        """The metadata fields `row()`/`row_by_num()` share — notably no `idx`,
        which is a TSV position and so only `row()` can answer.

        `genre` is `tags[0]`, and the TSV lists a track's tags in ALPHABETICAL
        order — so it is "the alphabetically first genre", not a primary one.
        `genres` carries the track's whole genre tag set for callers that need an
        honest label (see test_knn)."""
        tags = t["tags"]
        return {
            "track_id": t["track_id"],
            "artist_id": t["artist_id"],
            "album_id": t["album_id"],
            "title": t.get("title", ""),
            "artist": t.get("artist", ""),
            "album": t.get("album", ""),
            "released": t.get("released", None),
            "genre": tags["genre"][0] if tags.get("genre") else None,
            "genres": sorted(tags.get("genre", [])),
            "instrument": tags["instrument"][0] if tags.get("instrument") else None,
            "mood_theme": tags["mood/theme"][0] if tags.get("mood/theme") else None,
            "duration": t["duration"],
            "relpath": t["relpath"],
        }

    def row(self, idx: int) -> dict:
        """Metadata for track idx (no audio decode); `idx` is its TSV position."""
        return {"idx": idx, **self._fields(self._tracks[idx])}

    def row_by_num(self, num: int) -> dict | None:
        """Metadata for MTG track number `num` (the melspec `.npy` stem), or None.

        Carries no `idx`: a track number is not a position, and the caller owns
        whatever ordering it happens to be embedding in.
        """
        idx = self._num_to_idx.get(num)
        if idx is not None:
            return self._fields(self._tracks[idx])
        t = self._fallback.get(num)
        return None if t is None else self._fields(t)

    def __getitem__(self, idx: int) -> dict:
        return {**self.row(idx), "audio": AudioDecoder(str(self.audio_root / self._tracks[idx]["relpath"]))}
