"""Language score (y^text): labeling functions, LabelModel, and its diagnostics.

Outputs (under --out):
    units_language_scores.csv    meeting-level time ratios and scores
    zero_rates_and_guards.csv    zero rate and opportunity guard per indicator
    ablation.csv                 leave-one-subcategory-out correlations
    fig_language_score_distribution.png

Example:
    python 05_language_score.py --lang en
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import nts_common as nts

warnings.filterwarnings("ignore")


def main():
    ap = nts.add_common_args(argparse.ArgumentParser())
    args = ap.parse_args()
    L = nts.get_labels(args.lang)
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    CAT, TXT = L["category"], L["text"]

    df = nts.load_turns(args.turns)
    print(f"Utterances {len(df)} / meetings {df.meeting_id.nunique()}")

    df, units = nts.build_units(df)
    print(f"Meetings: {len(units)}")
    print("\nZero rate of each time ratio")
    print((units[nts.RATIO_COLS] == 0).mean().round(3).sort_values().to_string())

    units, cs, cat_L = nts.language_score(units)
    print("\nWeak score by subcategory")
    print(cs.rename(columns=CAT).describe().round(3).T[["mean", "50%", "std", "min", "max"]].to_string())
    print("\nCombined y_weak (language score)")
    print(units.y_weak.describe().round(3).to_string())

    for cat in nts.CATEGORY:
        ab = (cat_L[cat] == nts.ABSTAIN).all(axis=1)
        print(f"  {CAT[cat]:<20} all LFs abstain: {ab.sum():3d}/{len(units)} ({ab.mean()*100:4.1f}%)")
    allab = np.all([(cat_L[c] == nts.ABSTAIN).all(axis=1) for c in nts.CATEGORY], axis=0)
    print(f"  {'all four abstain':<20}: {allab.sum()}/{len(units)} ({allab.mean()*100:.1f}%)")

    # --- distribution of the language score ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(units.y_weak, bins=25, range=(0, 1), color="#4a7fa5", edgecolor="white")
    ax.set_xlabel(TXT["language_score"])
    ax.set_ylabel(TXT["n_meetings"])
    fig.tight_layout()
    fig.savefig(out / "fig_language_score_distribution.png", dpi=150)
    plt.close(fig)

    # --- zero rate and opportunity guard per indicator ---
    rows = []
    for cat, cols in nts.CATEGORY.items():
        for k in cols:
            v = units[k].values
            g = nts.guard_for(v, units.n.values)
            cov = ((v == 0) & (units.n.values >= g)).mean()
            rows.append({
                TXT["subcategory"]: CAT[cat],
                TXT["indicator"]: L["indicator"][k],
                TXT["zero_rate"]: round((v == 0).mean(), 3),
                TXT["guard"]: TXT["guard_none"] if g == -np.inf
                              else f"{int(round(g))} {TXT['utterances']}",
                TXT["absence_cov"]: round(cov, 3),
            })
    zr = pd.DataFrame(rows)
    print("\n" + zr.to_string(index=False))
    zr.to_csv(out / "zero_rates_and_guards.csv", index=False)

    # --- leave-one-subcategory-out ablation ---
    full = units.y_weak.values
    rows = []
    for c in nts.CATEGORY:
        yk = cs.drop(columns=c).mean(axis=1).values
        rows.append({TXT["excluded"]: CAT[c],
                     TXT["pearson"]: round(float(np.corrcoef(full, yk)[0, 1]), 3)})
    abl = pd.DataFrame(rows).sort_values(TXT["pearson"])
    print("\n" + abl.to_string(index=False))
    spread = abl[TXT["pearson"]].max() - abl[TXT["pearson"]].min()
    print(f"Spread (max - min): {spread:.3f}")
    abl.to_csv(out / "ablation.csv", index=False)

    # --- correlations among the four subcategories ---
    r4, _ = spearmanr(units[list(nts.CATEGORY)])
    off = r4[np.triu_indices(4, 1)]
    print("\n" + pd.DataFrame(r4, index=[CAT[c] for c in nts.CATEGORY],
                              columns=[CAT[c] for c in nts.CATEGORY]).round(3).to_string())
    print(f"Off-diagonal |rho|: mean {np.abs(off).mean():.3f}  max {np.abs(off).max():.3f}")

    keep = ["meeting_id", "tot", "n"] + nts.RATIO_COLS + list(nts.CATEGORY) + ["y_weak"]
    units[keep].to_csv(out / "units_language_scores.csv", index=False)
    print("\nsaved:", out / "units_language_scores.csv")


if __name__ == "__main__":
    main()
