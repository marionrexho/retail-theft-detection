"""Cut a short review clip around each alert, with the person boxed."""

from __future__ import annotations

from pathlib import Path

import cv2


def write_clips(video_path: Path, alerts: list, clip_dir: Path,
                before_s: float = 4.0, after_s: float = 2.0) -> list[str]:
    if not alerts:
        return []
    clip_dir = Path(clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    paths = []
    for n, ev in enumerate(alerts, 1):
        start = max(0.0, ev.reach_started - before_s)
        end = ev.t + after_s
        path = clip_dir / f"alert_{n:02d}_{ev.kind}_{ev.t:07.1f}s.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * fps))
        x1, y1, x2, y2 = (int(v) for v in ev.box)
        label = f"possible {ev.kind} - person {ev.person_id} - please review"
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
            if t > end:
                break
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"{t:6.1f}s", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            writer.write(frame)
        writer.release()
        paths.append(str(path))
    cap.release()
    return paths
