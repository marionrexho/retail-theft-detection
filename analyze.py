"""Run the detector on one video folder and print what it found.

    python analyze.py videos/1aa878f1
    python analyze.py videos/1aa878f1 --no-clips --fps 5

Results go to outputs/<name>/events.json, with review clips in outputs/<name>/clips/.
"""

import argparse
from pathlib import Path

from theft.labels import find_video, load_zones
from theft.pipeline import DEFAULT_DET_MODEL, DEFAULT_POSE_MODEL, analyze_video
from theft.reach import Params


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("video_dir", type=Path)
    add_common_args(ap)
    return ap


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--fps", type=float, default=10.0,
                    help="frames per second to analyze (lower is faster; default 10)")
    ap.add_argument("--max-width", type=int, default=960, help="shrink frames to this width")
    ap.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    ap.add_argument("--det-model", default=DEFAULT_DET_MODEL)
    ap.add_argument("--device", default=None, help="cpu, cuda:0 or mps (default: pick automatically)")
    ap.add_argument("--no-clips", action="store_true", help="skip writing review clips")
    ap.add_argument("--dwell", type=float, default=Params.dwell_s, help="seconds in shelf zone")
    ap.add_argument("--follow", type=float, default=Params.follow_s, help="seconds to follow a hand")
    ap.add_argument("--waist", type=float, default=Params.waist_frac,
                    help="wrist-to-hip distance for 'pocketed', as a fraction of body height")


def params_from(args) -> Params:
    return Params(dwell_s=args.dwell, follow_s=args.follow, waist_frac=args.waist)


def run(video_dir: Path, args) -> dict:
    out_dir = Path("outputs") / video_dir.name
    return analyze_video(
        find_video(video_dir), load_zones(video_dir), out_dir,
        params=params_from(args), fps_target=args.fps, max_width=args.max_width,
        pose_model=args.pose_model, det_model=args.det_model, device=args.device,
        save_clips=not args.no_clips,
    )


def main() -> None:
    args = build_parser().parse_args()
    result = run(args.video_dir, args)
    alerts = [e for e in result["events"] if e["alert"]]
    print(f"\n{args.video_dir.name}: {result['duration_s']}s of video, analyzed in "
          f"{result['seconds_taken']}s on {result['device']}")
    print(f"{len(result['events'])} reaches, {len(alerts)} alert(s)")
    for e in result["events"]:
        flag = "ALERT" if e["alert"] else "     "
        print(f"  {flag}  {e['t']:7.1f}s  {e['kind']:9s} person {e['person_id']} ({e['wrist']} hand)")
    for c in result.get("clips", []):
        print(f"  clip: {c}")


if __name__ == "__main__":
    main()
