"""Copy downloaded videos into videos/<name>/video.mp4, one folder each.

    python add_videos.py "C:\\Users\\you\\Downloads\\Theft Data"

Folder names use the first 8 characters of the file name. The original name is
kept in source.txt so you can always trace a folder back to the shared drive.
"""

import shutil
import sys
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv"}


def main() -> None:
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    src = Path(sys.argv[1])
    files = [src] if src.is_file() else sorted(p for p in src.iterdir()
                                               if p.suffix.lower() in VIDEO_EXTS)
    if not files:
        print(f"No videos found in {src}")
        sys.exit(1)
    root = Path(__file__).parent / "videos"
    for f in files:
        name = f.stem[:8].lower()
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / f"video{f.suffix.lower()}"
        if target.exists():
            print(f"skip   {name}  (already added)")
            continue
        shutil.copy2(f, target)
        (dest / "source.txt").write_text(f.name + "\n", encoding="utf-8")
        print(f"added  {name}  <- {f.name}")
    print(f"\nNext: python label.py videos/{files[0].stem[:8].lower()}")


if __name__ == "__main__":
    main()
