"""Build the non-verbal score (y^nonverbal) for every meeting.

The score averages three acoustic indicators, each mapped onto [0, 1]:

  speech rate  absolute anchors taken from the literature (trapezoid scoring:
               6-7 characters/second is the ideal band, below 5 or above 9.5 scores 0)
  F0 variation percentile rank within this corpus, inverted-U shaped
               (the median is best; both extremes score 0)
  intensity    percentile rank within this corpus (higher is better)

F0 and intensity use corpus-internal percentiles rather than absolute values:
F0 depends heavily on the individual speaker, and the recordings were not made under
uniform conditions, so published absolute thresholds do not transfer.
F0 variation is averaged per speaker first, so that the mix of speakers in a meeting
does not leak into the meeting-level value.

A fourth indicator, the pause between turns, is written to the output as well but is
NOT part of y_nonverbal. The ASR and diarization timestamps are too coarse: in many
meetings the gap at speaker changes rounds to almost zero. It is kept only as an
exploratory column (y_nonverbal_with_pause).

Example:
    python 04_nonverbal_scores.py
"""

import argparse
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

HERE = Path(__file__).resolve().parent
DEFAULT_IN = HERE.parent / "data" / "transcripts" / "nts_turns_master.csv"
DEFAULT_OUT = HERE.parent / "data" / "derived" / "nonverbal_scores.csv"


def trapezoid_score(x: pd.Series, ideal_low, ideal_high, floor_low, floor_high) -> pd.Series:
    """Score 1 inside [ideal_low, ideal_high], falling linearly to 0 at the floors.

    Values below floor_low or above floor_high score 0. The thresholds are the
    reference values reported in the literature, used directly as anchors.
    """
    x = x.astype(float)
    score = pd.Series(0.0, index=x.index)
    score[(x >= ideal_low) & (x <= ideal_high)] = 1.0
    rising = (x >= floor_low) & (x < ideal_low)
    score[rising] = (x[rising] - floor_low) / (ideal_low - floor_low)
    falling = (x > ideal_high) & (x <= floor_high)
    score[falling] = 1.0 - (x[falling] - ideal_high) / (floor_high - ideal_high)
    return score.clip(0.0, 1.0)


def percentile_rank(x: pd.Series) -> pd.Series:
    """Percentile rank in [0, 1]; monotonic, for indicators where higher is better."""
    return x.rank(pct=True)


def inverted_u_percentile_score(x: pd.Series) -> pd.Series:
    """1 at the median, falling to 0 at both ends of the corpus-internal distribution."""
    r = x.rank(pct=True)
    return 1.0 - 2.0 * (r - 0.5).abs()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    df = pd.read_csv(args.turns)
    df["text"] = df["text"].fillna("")
    # Utterances whose speaker could not be resolved are excluded throughout the study.
    df = df[df["speaker"] != "UNKNOWN"].copy()

    # --- meeting-level values of the two indicators taken straight from the turns ---
    mtg = (df.groupby("meeting_id")
             .agg(avg_speech_rate=("speech_rate_char_per_sec", "mean"),
                  rms_mean=("rms_mean", "mean"))
             .reset_index())

    # --- F0 variation: average per speaker first, then across speakers ---
    f0_by_speaker = (df.groupby(["meeting_id", "speaker"])["f0_std"]
                       .mean()
                       .reset_index(name="f0_std_per_speaker"))
    f0_speaker_avg = (f0_by_speaker.groupby("meeting_id")["f0_std_per_speaker"]
                                   .mean()
                                   .reset_index(name="f0_std_speaker_avg"))

    # --- pause between turns, at speaker changes only (exploratory) ---
    df_gap = df.sort_values(["meeting_id", "start"]).copy()
    df_gap["next_speaker"] = df_gap.groupby("meeting_id")["speaker"].shift(-1)
    turn_change = (df_gap["speaker"] != df_gap["next_speaker"]) & df_gap["next_speaker"].notna()
    sub_gap = df_gap[turn_change & df_gap["gap_to_next"].notna()]
    gap_agg = (sub_gap.groupby("meeting_id")["gap_to_next"]
                      .agg(mean_gap="mean",
                           median_gap="median",
                           long_pause_ratio=lambda s: (s > 1.5).mean())
                      .reset_index())

    n_missing = mtg["meeting_id"].nunique() - gap_agg["meeting_id"].nunique()
    n_zero_median = (gap_agg["median_gap"] == 0).sum()
    print(f"Meetings without a single speaker-change gap: {n_missing} / {mtg['meeting_id'].nunique()}")
    print(f"Meetings whose median gap is 0 s: {n_zero_median} / {len(gap_agg)}")

    # --- scoring ---
    score_speechrate = trapezoid_score(
        mtg["avg_speech_rate"], ideal_low=6.0, ideal_high=7.0, floor_low=5.0, floor_high=9.5
    )
    score_f0var = inverted_u_percentile_score(f0_speaker_avg["f0_std_speaker_avg"])
    score_intensity = percentile_rank(mtg["rms_mean"])

    nv_scores = pd.DataFrame({
        "meeting_id": mtg["meeting_id"],
        "score_speechrate": score_speechrate.values,
        "score_intensity": score_intensity.values,
    })
    nv_scores = nv_scores.merge(
        f0_speaker_avg[["meeting_id"]].assign(score_f0var=score_f0var.values),
        on="meeting_id", how="left",
    )

    # Main score: the mean of the three indicators.
    nv_scores["y_nonverbal"] = nv_scores[
        ["score_speechrate", "score_f0var", "score_intensity"]
    ].mean(axis=1)

    # Exploratory variant that also includes the pause indicator.
    nv_scores = nv_scores.merge(gap_agg[["meeting_id", "mean_gap"]], on="meeting_id", how="left")
    nv_scores["mean_gap_filled"] = nv_scores["mean_gap"].fillna(nv_scores["mean_gap"].median())
    nv_scores["score_pause"] = trapezoid_score(
        nv_scores["mean_gap_filled"], ideal_low=0.1, ideal_high=0.3, floor_low=0.0, floor_high=4.0
    )
    nv_scores["y_nonverbal_with_pause"] = nv_scores[
        ["score_speechrate", "score_f0var", "score_intensity", "score_pause"]
    ].mean(axis=1)

    print(nv_scores[["y_nonverbal", "y_nonverbal_with_pause"]].describe().round(3).to_string())
    r, _ = pearsonr(nv_scores["y_nonverbal"], nv_scores["y_nonverbal_with_pause"])
    print(f"\nMain score (3 indicators) vs. variant with pause (4 indicators): r = {r:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nv_scores.to_csv(out_path, index=False)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
