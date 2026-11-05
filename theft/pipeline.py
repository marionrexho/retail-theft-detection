"""Run the models over a video and turn their output into events."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2

from .geometry import denormalize_polygons
from .reach import Params, Person, ReachTracker

# COCO classes from the standard YOLO detector that count as "a bag".
BAG_CLASSES = {24: "backpack", 26: "handbag", 28: "suitcase"}

DEFAULT_POSE_MODEL = "yolo11n-pose.pt"   # small and fast; fine on a laptop CPU
DEFAULT_DET_MODEL = "yolo11n.pt"


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def analyze_video(
    video_path: Path,
    zones_norm: list,
    out_dir: Path,
    params: Params | None = None,
    fps_target: float = 10.0,
    max_width: int = 960,
    bag_every: int = 3,
    pose_model: str = DEFAULT_POSE_MODEL,
    det_model: str = DEFAULT_DET_MODEL,
    device: str | None = None,
    save_clips: bool = True,
    progress: bool = True,
) -> dict:
    """Analyze one video. Writes events.json (and clips/) into out_dir and returns the result."""
    from ultralytics import YOLO  # imported here so the rest of the package loads without it

    if not zones_norm:
        raise ValueError(
            f"No shelf zone for {video_path}. Draw one first with: python label.py {video_path}"
        )

    params = params or Params()
    device = pick_device(device)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = n_frames / src_fps if n_frames else 0.0

    scale = min(1.0, max_width / width) if width else 1.0
    w_s, h_s = int(width * scale), int(height * scale)
    zones = denormalize_polygons(zones_norm, w_s, h_s)
    stride = max(1, round(src_fps / fps_target))

    pose = YOLO(pose_model)
    det = YOLO(det_model)
    tracker = ReachTracker(params)

    events = []
    bags: list = []
    frame_idx = processed = 0
    started = time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        if frame_idx % stride:
            frame_idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = frame_idx / src_fps
        if scale < 1.0:
            frame = cv2.resize(frame, (w_s, h_s), interpolation=cv2.INTER_AREA)

        if processed % bag_every == 0:
            res = det.predict(frame, classes=list(BAG_CLASSES), conf=0.25,
                              device=device, verbose=False)[0]
            bags = [tuple(b) for b in res.boxes.xyxy.cpu().numpy().tolist()] if res.boxes else []

        res = pose.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                         conf=0.3, device=device, verbose=False)[0]
        people = _people_from_result(res)

        for ev in tracker.update(t, people, bags, zones):
            ev.box = tuple(v / scale for v in ev.box)  # back to original pixels
            events.append(ev)

        processed += 1
        frame_idx += 1
        if progress and processed % 50 == 0:
            pct = f"{100 * frame_idx / n_frames:5.1f}%" if n_frames else ""
            print(f"  {video_path.name}: {pct} ({time.time() - started:.0f}s)", flush=True)
    cap.release()

    result = {
        "video": str(video_path),
        "duration_s": round(duration, 2),
        "fps_analyzed": round(src_fps / stride, 2),
        "device": device,
        "models": {"pose": pose_model, "detector": det_model},
        "params": asdict(params),
        "seconds_taken": round(time.time() - started, 1),
        "events": [e.to_dict() for e in events],
    }
    if save_clips:
        from .clips import write_clips
        result["clips"] = write_clips(video_path, [e for e in events if e.alert], out_dir / "clips")
    (out_dir / "events.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _people_from_result(res) -> list[Person]:
    if res.boxes is None or res.boxes.id is None or res.keypoints is None:
        return []
    ids = res.boxes.id.int().cpu().tolist()
    boxes = res.boxes.xyxy.cpu().numpy().tolist()
    kps = res.keypoints.data.cpu().numpy().tolist()  # (n, 17, 3): x, y, conf
    return [Person(track_id=i, box=tuple(b), keypoints=k) for i, b, k in zip(ids, boxes, kps)]
