"""Label a video and draw its shelf zone.

    python label.py videos/1aa878f1

A window opens with the video. Keys:

    space        play / pause
    a / d        back / forward 2 seconds      j / l   back / forward 10 seconds
    , / .        one frame back / forward

    1  bagged      2  pocketed     3  concealed      (theft)
    4  carted      5  put_back     6  own_item       (normal)
    u            undo the last label
    r            remove the label nearest the current time (within 1 s)

    z            draw a shelf zone: click the corners, then press Enter
    x            clear all shelf zones
    h            show / hide this help
    s            save        q   save and quit

Label the moment the item goes into the bag, pocket, cart, or back on the shelf.
"""

import sys
from pathlib import Path

import cv2

from theft.labels import (ALL_LABELS, THEFT_LABELS, find_video, load_labels, load_zones,
                          save_labels, save_zones)

KEY_LABELS = {ord(str(i + 1)): lab for i, lab in enumerate(ALL_LABELS)}
WINDOW = "label - press h for help"
MAX_W, MAX_H = 1280, 720
RED, GREEN, YELLOW, WHITE = (60, 60, 230), (80, 200, 80), (0, 220, 255), (255, 255, 255)

HELP = [
    "space play/pause   a/d -/+2s   j/l -/+10s   ,/. frame",
    "1 bagged  2 pocketed  3 concealed   (theft)",
    "4 carted  5 put_back  6 own_item    (normal)",
    "u undo   r remove nearest   z draw zone   x clear zones",
    "s save   q save & quit   h hide help",
]


class Labeler:
    def __init__(self, video_dir: Path):
        self.dir = video_dir
        self.video = find_video(video_dir)
        self.cap = cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            sys.exit(f"Could not open {self.video}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        s = min(1.0, MAX_W / w, MAX_H / h)
        self.size = (int(w * s), int(h * s))
        self.data = load_labels(video_dir)
        self.zones = load_zones(video_dir)
        self.added: list[dict] = []
        self.drawing: list | None = None
        self.playing = False
        self.show_help = True
        self.pos = 0
        self.frame = None
        self.dirty = False
        self.seek(0)

    # --- video -----------------------------------------------------------
    def seek(self, frame_no: int) -> None:
        self.pos = max(0, min(self.n - 1, frame_no))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.pos)
        self.read()

    def read(self) -> bool:
        ok, frame = self.cap.read()
        if not ok:
            self.playing = False
            return False
        self.pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        self.frame = cv2.resize(frame, self.size)
        return True

    @property
    def t(self) -> float:
        return self.pos / self.fps

    # --- labels ----------------------------------------------------------
    def add_label(self, label: str) -> None:
        ev = {"t": round(self.t, 2), "label": label}
        self.data["events"].append(ev)
        self.added.append(ev)
        self.dirty = True
        print(f"  {ev['t']:7.2f}s  {label}")

    def undo(self) -> None:
        if self.added:
            ev = self.added.pop()
            self.data["events"].remove(ev)
            self.dirty = True
            print(f"  removed {ev['t']:.2f}s {ev['label']}")

    def remove_nearest(self) -> None:
        evs = self.data["events"]
        if not evs:
            return
        ev = min(evs, key=lambda e: abs(e["t"] - self.t))
        if abs(ev["t"] - self.t) <= 1.0:
            evs.remove(ev)
            if ev in self.added:
                self.added.remove(ev)
            self.dirty = True
            print(f"  removed {ev['t']:.2f}s {ev['label']}")

    def save(self) -> None:
        save_labels(self.dir, self.data)
        save_zones(self.dir, self.zones)
        self.dirty = False
        print(f"  saved {len(self.data['events'])} labels, {len(self.zones)} zone(s)")

    # --- drawing ---------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param) -> None:
        if self.drawing is not None and event == cv2.EVENT_LBUTTONDOWN:
            self.drawing.append((x / self.size[0], y / self.size[1]))

    def render(self):
        img = self.frame.copy()
        w, h = self.size
        for poly in self.zones:
            pts = [(int(x * w), int(y * h)) for x, y in poly]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                cv2.line(img, a, b, YELLOW, 2)
        if self.drawing:
            pts = [(int(x * w), int(y * h)) for x, y in self.drawing]
            for p in pts:
                cv2.circle(img, p, 4, YELLOW, -1)
            for a, b in zip(pts, pts[1:]):
                cv2.line(img, a, b, YELLOW, 1)

        status = "PLAYING" if self.playing else "PAUSED"
        if self.drawing is not None:
            status = "DRAWING ZONE: click corners, Enter to finish, Esc to cancel"
        _text(img, f"{self.t:7.2f}s / {self.n / self.fps:.1f}s   {status}", (10, h - 15))

        near = [e for e in self.data["events"] if abs(e["t"] - self.t) <= 3]
        for i, e in enumerate(near):
            colour = RED if e["label"] in THEFT_LABELS else GREEN
            _text(img, f"{e['t']:.1f}s {e['label']}", (w - 220, 25 + 22 * i), colour)

        if self.show_help:
            for i, line in enumerate(HELP):
                _text(img, line, (10, 25 + 22 * i))
        # A thin bar showing every label along the timeline.
        for e in self.data["events"]:
            x = int(w * e["t"] * self.fps / max(1, self.n))
            cv2.line(img, (x, h - 4), (x, h), RED if e["label"] in THEFT_LABELS else GREEN, 3)
        return img

    # --- main loop -------------------------------------------------------
    def run(self) -> None:
        cv2.namedWindow(WINDOW)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        step2, step10 = int(2 * self.fps), int(10 * self.fps)
        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(max(1, int(1000 / self.fps)) if self.playing else 30) & 0xFF
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
            if self.playing:
                self.read()

            if self.drawing is not None:
                if key in (13, 10):  # Enter
                    if len(self.drawing) >= 3:
                        self.zones.append(self.drawing)
                        self.dirty = True
                    self.drawing = None
                elif key == 27:  # Esc
                    self.drawing = None
                continue

            if key == ord(" "):
                self.playing = not self.playing
            elif key == ord("a"):
                self.seek(self.pos - step2)
            elif key == ord("d"):
                self.seek(self.pos + step2)
            elif key == ord("j"):
                self.seek(self.pos - step10)
            elif key == ord("l"):
                self.seek(self.pos + step10)
            elif key == ord(","):
                self.seek(self.pos - 1)
            elif key == ord("."):
                self.read()
            elif key in KEY_LABELS:
                self.add_label(KEY_LABELS[key])
            elif key == ord("u"):
                self.undo()
            elif key == ord("r"):
                self.remove_nearest()
            elif key == ord("z"):
                self.playing = False
                self.drawing = []
            elif key == ord("x"):
                self.zones = []
                self.dirty = True
            elif key == ord("h"):
                self.show_help = not self.show_help
            elif key == ord("s"):
                self.save()
            elif key == ord("q"):
                break
        cv2.destroyAllWindows()
        self.cap.release()
        self.finish()

    def finish(self) -> None:
        print()
        camera = input(f"Camera name [{self.data.get('camera', '')}]: ").strip()
        if camera:
            self.data["camera"] = camera
        note = input(f"One-line note [{self.data.get('note', '')}]: ").strip()
        if note:
            self.data["note"] = note
        done = input("Did you watch the whole video and label everything? (y/n): ")
        self.data["done"] = done.strip().lower().startswith("y")
        self.save()
        if not self.zones:
            print("  Warning: no shelf zone drawn. The detector needs one (press z).")


def _text(img, text, org, colour=WHITE):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    Labeler(Path(sys.argv[1])).run()
