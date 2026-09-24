"""Collect the per-meeting pipeline output into a single utterance-level table.

Reads every tvNNN/turns_with_text_and_acoustics.csv produced by 01_audio_pipeline.py,
adds the meeting id and two meeting-level fields from meeting_summary.json, and writes
data/transcripts/nts_turns_master.csv, which is the input to every analysis script.

Example:
    python 03_build_turns_master.py --meeting_output ../data/meeting_output
"""

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE.parent / "data" / "transcripts" / "nts_turns_master.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meeting_output", required=True,
                    help="root directory holding the per-meeting output folders")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    root = Path(args.meeting_output)
    frames = []

    for tv_dir in sorted(root.glob("tv*")):
        turns_path = tv_dir / "turns_with_text_and_acoustics.csv"
        summary_path = tv_dir / "meeting_summary.json"

        if not turns_path.exists():
            print(f"[WARN] {turns_path} not found; skipping this meeting.")
            continue

        df = pd.read_csv(turns_path)
        df["meeting_id"] = tv_dir.name

        # Meeting-level fields, repeated on every row of the meeting.
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            df["meeting_duration_sec"] = summary.get("duration_sec", None)
            df["silence_ratio"] = summary.get("silence_ratio", None)
        else:
            df["meeting_duration_sec"] = None
            df["silence_ratio"] = None

        frames.append(df)

    if not frames:
        raise SystemExit(f"No meeting folders found under {root}")

    all_turns = pd.concat(frames, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_turns.to_csv(out_path, index=False)

    print("saved:", out_path)
    print("rows:", len(all_turns))
    print("meetings:", all_turns["meeting_id"].nunique())


if __name__ == "__main__":
    main()
