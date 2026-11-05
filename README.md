# Retail Theft Detection

Flags moments in store camera footage when someone takes an item from a shelf and puts it into their own bag or pocket, then cuts a short clip for a person to review. It never decides that someone stole; it points a human at a clip worth checking.

It comes with its own labeling tool and a scoring script, so every change can be measured against real footage instead of guessed at.

## How it works

```
video ─► track people and their wrists ─► wrist stays in the shelf zone ≥ 0.2 s
      ─► follow the wrist for 3 s ─► where does it go first?
             bag the person carries ──► "bagged"   (alert + clip)
             the person's own waist ──► "pocketed" (alert + clip)
             back to the shelf ───────► cancelled  (put back, or reaching again)
             nowhere in time ─────────► "unknown"  (logged, no alert)
```

- **People and wrists:** a YOLO pose model finds each person's body points; ByteTrack keeps the same ID on a person across frames.
- **Bags:** a standard YOLO detector finds backpacks, handbags and suitcases. A bag only counts if it overlaps the person.
- **Shelf zone:** drawn by hand once per camera. A reach only starts inside it.
- **No item detection:** the system never tries to see the product, which is often only a few pixels wide on CCTV. It follows the hand instead.

The decision logic lives in [`theft/reach.py`](theft/reach.py) and has no model code in it, so it is fully unit tested ([`tests/`](tests/)).

## Setup (Windows)

1. Install Python 3.12 if you don't have 3.10, 3.11 or 3.12. Newer versions are fine to keep, but the AI libraries don't support them yet:
   ```
   winget install Python.Python.3.12
   ```
2. Clone this repo and double-click **`setup_windows.bat`**. It creates a `venv` folder, installs everything (CPU version, about 1 GB), and runs the tests.
3. For every session, open a command window in the project folder and run:
   ```
   venv\Scripts\activate
   ```

No GPU is needed. With one, the scripts use it automatically.

## Using it

**1. Add videos.** Download the footage from the shared drive, then copy it into the project:
```
python add_videos.py "C:\Users\you\Downloads\Theft Data"
```
Each video gets a folder like `videos/1aa878f1/`.

**2. Label a video and draw its shelf zone:**
```
python label.py videos/1aa878f1
```
| Key | Does |
| --- | --- |
| `space` | play / pause |
| `a` `d` / `j` `l` / `,` `.` | back/forward 2 s / 10 s / 1 frame |
| `1` `2` `3` | **theft:** bagged, pocketed, concealed |
| `4` `5` `6` | **normal:** carted, put_back, own_item |
| `u` / `r` | undo last label / remove the label nearest now |
| `z` | draw a shelf zone: click its corners, press Enter |
| `x` | clear zones |
| `s` / `q` | save / save and quit |

Label the moment the item goes in (bag, pocket, cart, back on the shelf). When you quit, it asks for the camera name, a note, and whether you watched the whole video. Only videos you mark as fully watched are scored.

**3. Run the detector on one video:**
```
python analyze.py videos/1aa878f1
```
Results go to `outputs/1aa878f1/`: `events.json` and a review clip per alert in `clips/`.

**4. Score it against every labeled video:**
```
python evaluate.py
```
```
Catch rate:            67% (4 of 6 thefts)
Alerts that were real: 50%
False alarms per hour: 12.0 (4 in 20.0 min of video)
```
(example output) Full details go to `results/report.md`, with a report per video in `outputs/<name>/report.md`, including which normal action caused each false alarm. Add `--reuse` to re-score without re-running the models.

## Tuning

Change one thing, run `evaluate.py`, and keep the change only if the scores improve. Useful options for `analyze.py` and `evaluate.py`:

| Option | Default | Effect |
| --- | --- | --- |
| `--fps` | 10 | Frames analyzed per second. Lower is faster on a CPU. |
| `--dwell` | 0.2 | Seconds a hand must spend in the shelf zone. |
| `--follow` | 3.0 | Seconds to follow the hand after it leaves the shelf. |
| `--waist` | 0.12 | How close to the hip counts as "pocketed" (fraction of body height). |
| `--pose-model` | yolo11n-pose.pt | Try `yolo11s-pose.pt` for accuracy at the cost of speed. |

## Data rules

The footage shows real people. **Videos and clips are never committed** (`.gitignore` blocks them). Only `labels.json` and `zones.json` go in the repo.

## Known limits

- Tested on a handful of clips so far; settings may be tuned to them.
- Hand position alone can't tell pocketing from touching a phone or adjusting clothes near the waist.
- Concealing items in a jacket, or a bag the detector misses, is not detected yet.
- Shopping carts and baskets are not detected, so a hand going to a cart ends as "unknown" rather than "carted".
- Works on video files, not live camera streams yet.

## Project layout

```
add_videos.py   copy footage into videos/<name>/
label.py        labeling tool (answer key + shelf zone)
analyze.py      run the detector on one video
evaluate.py     score the detector against all labeled videos
setup_windows.bat
theft/
  reach.py      decision logic (shelf → bag / pocket)
  pipeline.py   runs the models over a video
  clips.py      cuts review clips
  scoring.py    matches alerts to labels
  labels.py     label and zone file formats
  geometry.py   point-in-polygon and box helpers
tests/          unit tests (run: python -m pytest)
videos/<name>/  labels.json, zones.json (video.mp4 stays local)
```
