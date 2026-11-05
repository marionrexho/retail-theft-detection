"""Grade the detector against every labeled video.

    python evaluate.py              run the detector on each labeled video, then score
    python evaluate.py --reuse      score saved results without re-running (fast)

Only videos marked done (you watched the whole thing) are scored.
Prints the overall score and writes results/report.md plus a report per video.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from analyze import add_common_args, run
from theft.labels import load_labels, load_zones
from theft.scoring import VideoScore, score_video, summarize


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--reuse", action="store_true", help="use saved outputs/<name>/events.json")
    ap.add_argument("--tolerance", type=float, default=3.0,
                    help="seconds an alert may be off from the label and still count")
    ap.add_argument("--only", nargs="*", help="only these video folder names")
    add_common_args(ap)
    args = ap.parse_args()

    scores: list[VideoScore] = []
    skipped = []
    for vdir in sorted(p for p in Path("videos").iterdir() if p.is_dir()):
        if args.only and vdir.name not in args.only:
            continue
        labels = load_labels(vdir)
        if not labels["done"]:
            skipped.append((vdir.name, "not marked done"))
            continue
        if not load_zones(vdir):
            skipped.append((vdir.name, "no shelf zone"))
            continue
        saved = Path("outputs") / vdir.name / "events.json"
        if args.reuse and saved.exists():
            result = json.loads(saved.read_text(encoding="utf-8"))
        else:
            print(f"Analyzing {vdir.name} ...", flush=True)
            try:
                result = run(vdir, args)
            except FileNotFoundError as e:
                skipped.append((vdir.name, str(e).split(".")[0]))
                continue
        alerts = [e for e in result["events"] if e["alert"]]
        s = score_video(vdir.name, result["duration_s"], labels["events"], alerts, args.tolerance)
        scores.append(s)
        write_video_report(vdir, s, result, labels)

    if not scores:
        print("Nothing to score. Label at least one video and mark it done (python label.py ...).")
        for name, why in skipped:
            print(f"  skipped {name}: {why}")
        return

    summary = summarize(scores)
    print_summary(scores, summary, skipped)
    write_report(scores, summary, skipped, args)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.0f}%"


def print_summary(scores, summary, skipped) -> None:
    print()
    print(f"{'video':10s} {'thefts':>6s} {'caught':>6s} {'missed':>6s} {'false':>6s}")
    for s in scores:
        print(f"{s.name:10s} {s.n_thefts:6d} {len(s.caught):6d} {len(s.missed):6d} "
              f"{len(s.false_alarms):6d}")
    print()
    print(f"Catch rate:            {pct(summary['catch_rate'])} "
          f"({summary['caught']} of {summary['thefts_labeled']} thefts)")
    print(f"Alerts that were real: {pct(summary['precision'])}")
    fph = summary["false_alarms_per_hour"]
    print(f"False alarms per hour: {'n/a' if fph is None else f'{fph:.1f}'} "
          f"({summary['false_alarms']} in {summary['hours_of_video'] * 60:.1f} min of video)")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    print("\nDetails: results/report.md")


def write_video_report(vdir: Path, s: VideoScore, result: dict, labels: dict) -> None:
    lines = [f"# {s.name}", "", f"Camera: {labels.get('camera') or '-'}  ",
             f"Note: {labels.get('note') or '-'}  ", f"Length: {s.duration_s:.0f}s", ""]
    lines += ["## Caught", ""] + ([f"- {l['t']:.1f}s {l['label']} -> alert at {a['t']:.1f}s "
                                   f"({a['kind']}, person {a['person_id']})" for l, a in s.caught]
                                  or ["- none"])
    lines += ["", "## Missed", ""] + ([f"- {l['t']:.1f}s {l['label']}" for l in s.missed]
                                      or ["- none"])
    lines += ["", "## False alarms", ""]
    lines += [f"- {a['t']:.1f}s {a['kind']}, person {a['person_id']}"
              + (f" (near a labeled {n['label']} at {n['t']:.1f}s)" if n else "")
              for a, n in s.false_alarms] or ["- none"]
    lines += ["", "## Everything the detector saw", ""]
    lines += [f"- {e['t']:.1f}s {e['kind']}{' (alert)' if e['alert'] else ''}, "
              f"person {e['person_id']}, {e['wrist']} hand" for e in result["events"]] or ["- none"]
    out = Path("outputs") / s.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(scores, summary, skipped, args) -> None:
    Path("results").mkdir(exist_ok=True)
    fph = summary["false_alarms_per_hour"]
    lines = [
        "# Evaluation report", "",
        f"Run: {datetime.now():%Y-%m-%d %H:%M}  ",
        f"Settings: fps={args.fps}, dwell={args.dwell}s, follow={args.follow}s, "
        f"waist={args.waist}, tolerance={args.tolerance}s, pose={args.pose_model}", "",
        "| Measure | Value |", "| --- | --- |",
        f"| Catch rate | {pct(summary['catch_rate'])} ({summary['caught']} of "
        f"{summary['thefts_labeled']}) |",
        f"| Alerts that were real | {pct(summary['precision'])} |",
        f"| False alarms per hour | {'n/a' if fph is None else f'{fph:.1f}'} |",
        f"| Video scored | {summary['hours_of_video'] * 60:.1f} min in {summary['videos']} videos |",
        "", "| Video | Thefts | Caught | Missed | False alarms |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [f"| [{s.name}](../outputs/{s.name}/report.md) | {s.n_thefts} | {len(s.caught)} | "
              f"{len(s.missed)} | {len(s.false_alarms)} |" for s in scores]
    if skipped:
        lines += ["", "Skipped: " + ", ".join(f"{n} ({w})" for n, w in skipped)]
    (Path("results") / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
