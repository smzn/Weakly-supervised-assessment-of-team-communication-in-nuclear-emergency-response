"""Run the audio pipeline over every meeting recording.

The recordings themselves are not part of this repository (see data/README.md).
Place one WAV file per meeting in the input directory, named tv001.wav ... tv161.wav,
and this script calls 01_audio_pipeline.py once per file.

Example:
    python 02_run_pipeline_batch.py --audio_dir /path/to/wav --outdir ../data/meeting_output
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "01_audio_pipeline.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="directory holding tvNNN.wav files")
    ap.add_argument("--outdir", default=str(HERE.parent / "data" / "meeting_output"),
                    help="root directory for the per-meeting output folders")
    ap.add_argument("--first", type=int, default=1)
    ap.add_argument("--last", type=int, default=161)
    ap.add_argument("--language", default="ja")
    ap.add_argument("--whisper_model", default="large-v3")
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_root = Path(args.outdir)
    n_done, n_skipped = 0, 0

    for i in range(args.first, args.last + 1):
        meeting_id = f"tv{i:03d}"
        audio_path = audio_dir / f"{meeting_id}.wav"
        if not audio_path.exists():
            print(f"Skip: {audio_path} not found.")
            n_skipped += 1
            continue

        print(f"Running: {meeting_id}")
        cmd = [
            sys.executable, str(PIPELINE),
            "--audio", str(audio_path),
            "--outdir", str(out_root / meeting_id),
            "--language", args.language,
            "--whisper_model", args.whisper_model,
        ]
        subprocess.run(cmd, check=False)
        n_done += 1

    print(f"Processed {n_done} meetings, skipped {n_skipped}.")


if __name__ == "__main__":
    main()
