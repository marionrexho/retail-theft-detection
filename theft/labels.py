"""Label and zone file formats.

videos/<name>/
    video.mp4      the footage (never committed)
    labels.json    what really happens, written by a person (the answer key)
    zones.json     shelf areas, as 0-1 fractions of the frame
"""

from __future__ import annotations

import json
from pathlib import Path

# Labels a person can give an event. Theft labels should raise an alert.
THEFT_LABELS = ["bagged", "pocketed", "concealed"]
NORMAL_LABELS = ["carted", "put_back", "own_item"]
ALL_LABELS = THEFT_LABELS + NORMAL_LABELS

# What the detector outputs, and which of those are alerts.
ALERT_KINDS = {"bagged", "pocketed"}


VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


def find_video(video_dir: Path) -> Path:
    """The footage inside a video folder (video.mp4, or any single video file)."""
    video_dir = Path(video_dir)
    if video_dir.is_file():
        return video_dir
    for ext in VIDEO_EXTS:
        p = video_dir / f"video{ext}"
        if p.exists():
            return p
    others = [p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS]
    if len(others) == 1:
        return others[0]
    raise FileNotFoundError(
        f"No video in {video_dir}. Get it from the shared drive and save it as "
        f"{video_dir / 'video.mp4'}"
    )


def labels_path(video_dir: Path) -> Path:
    return Path(video_dir) / "labels.json"


def zones_path(video_dir: Path) -> Path:
    return Path(video_dir) / "zones.json"


def load_labels(video_dir: Path) -> dict:
    p = labels_path(video_dir)
    if not p.exists():
        return {"camera": "", "note": "", "done": False, "events": []}
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("events", [])
    data.setdefault("done", False)
    for ev in data["events"]:
        if ev["label"] not in ALL_LABELS:
            raise ValueError(f"{p}: unknown label {ev['label']!r}")
    return data


def save_labels(video_dir: Path, data: dict) -> None:
    data["events"] = sorted(data.get("events", []), key=lambda e: e["t"])
    labels_path(video_dir).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_zones(video_dir: Path) -> list[list[tuple[float, float]]]:
    p = zones_path(video_dir)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return [[(float(x), float(y)) for x, y in poly] for poly in data.get("zones", [])]


def save_zones(video_dir: Path, zones: list) -> None:
    payload = {"zones": [[[round(x, 4), round(y, 4)] for x, y in poly] for poly in zones]}
    zones_path(video_dir).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
