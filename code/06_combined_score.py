"""weak COMM score (y^COMM): combine the language and non-verbal axes.

The weight alpha comes from the Nash bargaining solution. Each axis is a player whose
utility is the correlation between the combined score and that axis; the disagreement
point is the correlation between the two axes themselves. The solution maximises the
product of the gains over that point.

Outputs (under --out):
    units_with_scores.csv        meeting-level scores used by 07_keyword_analysis.py
    nash_bargaining_curve.csv    the Nash product for alpha from 0 to 1
    fig_nonverbal_score_distribution.png
    fig_language_vs_nonverbal.png
    fig_nash_product_and_axes.png

Example:
    python 06_combined_score.py --lang en
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler

import nts_common as nts

warnings.filterwarnings("ignore")


def main():
    ap = nts.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--nonverbal", default=str(nts.DEFAULT_NONVERBAL))
    ap.add_argument("--language_scores", default=None,
                    help="units_language_scores.csv; recomputed from the turns if omitted")
    args = ap.parse_args()
    L = nts.get_labels(args.lang)
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    TXT = L["text"]

    if args.language_scores and Path(args.language_scores).exists():
        units = pd.read_csv(args.language_scores)
    else:
        df = nts.load_turns(args.turns)
        _, units = nts.build_units(df)
        units, _, _ = nts.language_score(units)

    nv = pd.read_csv(args.nonverbal)
    m = units.merge(nv[["meeting_id", "y_nonverbal"]], on="meeting_id", how="inner")
    print(f"Meetings after merge: {len(m)}")

    r_axes, p_axes = pearsonr(m.y_weak, m.y_nonverbal)
    rs, ps = spearmanr(m.y_weak, m.y_nonverbal)
    print(f"Language x non-verbal  Pearson r = {r_axes:.4f} (p = {p_axes:.3g})")
    print(f"  Reference: Spearman rho = {rs:.4f} (p = {ps:.3g})")

    # --- distribution of the non-verbal score ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(m.y_nonverbal, bins=25, range=(0, 1), color="#4a7fa5", edgecolor="white")
    ax.set_xlabel(TXT["nonverbal_score"])
    ax.set_ylabel(TXT["n_meetings"])
    fig.tight_layout()
    fig.savefig(out / "fig_nonverbal_score_distribution.png", dpi=150)
    plt.close(fig)

    # --- Nash bargaining solution ---
    y_w, y_nv = m.y_weak.values, m.y_nonverbal.values
    d = r_axes
    print(f"\nDisagreement point d = corr(y_weak, y_nonverbal) = {d:.4f}")

    alphas = np.linspace(0.0, 1.0, 1001)
    rows = []
    for a in alphas:
        yc = a * y_w + (1 - a) * y_nv
        ut, _ = pearsonr(yc, y_w)
        uv, _ = pearsonr(yc, y_nv)
        gt, gv = ut - d, uv - d
        rows.append((a, ut, uv, gt * gv if (gt > 0 and gv > 0) else np.nan))
    nash = pd.DataFrame(rows, columns=["alpha", "u_text", "u_voice", "nash_product"])

    bi = nash.nash_product.idxmax()
    alpha_nash = float(nash.loc[bi, "alpha"])
    beta_nash = 1 - alpha_nash
    print(f"Nash bargaining solution: alpha = {alpha_nash:.4f}, beta = {beta_nash:.4f}")
    print(f"  u_text = {nash.loc[bi,'u_text']:.4f}, u_voice = {nash.loc[bi,'u_voice']:.4f}, "
          f"N = {nash.loc[bi,'nash_product']:.4f}")

    # Re-solving on standardized axes gives exactly 0.5; the raw solution is close to it
    # only because the two axes happen to have nearly the same spread in this corpus.
    Z = StandardScaler().fit_transform(np.c_[y_w, y_nv])
    zw, znv = Z[:, 0], Z[:, 1]
    dz, _ = pearsonr(zw, znv)
    prod_z = []
    for a in alphas:
        yc = a * zw + (1 - a) * znv
        ut, _ = pearsonr(yc, zw)
        uv, _ = pearsonr(yc, znv)
        gt, gv = ut - dz, uv - dz
        prod_z.append(gt * gv if (gt > 0 and gv > 0) else np.nan)
    alpha_z = float(alphas[int(np.nanargmax(prod_z))])
    print(f"Re-solved on standardized scores: alpha = {alpha_z:.4f}")
    print(f"  SD of y_weak {y_w.std():.4f} / SD of y_nonverbal {y_nv.std():.4f}")

    m["y_combined"] = alpha_nash * y_w + beta_nash * y_nv
    nash.to_csv(out / "nash_bargaining_curve.csv", index=False)
    print("\ny_combined")
    print(m.y_combined.describe().round(3).to_string())

    # --- scatter of the two axes, highlighting the top 20% of the language score ---
    thr = m.y_weak.quantile(0.80)
    is_top = m.y_weak >= thr
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(m.loc[~is_top, "y_weak"], m.loc[~is_top, "y_nonverbal"], s=26, alpha=.45,
               color="#8fa8bd", edgecolor="white", label=TXT["other_meetings"])
    ax.scatter(m.loc[is_top, "y_weak"], m.loc[is_top, "y_nonverbal"], s=34, alpha=.85,
               color="#e8963c", edgecolor="white", label=TXT["top20"])
    ax.set_xlabel(TXT["language_score"])
    ax.set_ylabel(TXT["nonverbal_score"])
    ax.set_title(TXT["scatter_title_fmt"].format(r=r_axes))
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig_language_vs_nonverbal.png", dpi=150)
    plt.close(fig)

    hi = m[is_top]
    print(f"\nTop 20% language score (threshold {thr:.3f}): {len(hi)} meetings")
    print(f"  Their non-verbal score range: {hi.y_nonverbal.min():.3f} to {hi.y_nonverbal.max():.3f}")

    # --- two-panel figure: the axes, and the Nash product ---
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].scatter(m.y_weak, m.y_nonverbal, s=26, alpha=.65,
                  color="#4a7fa5", edgecolor="white")
    ax[0].set_xlabel(TXT["language_score"])
    ax[0].set_ylabel(TXT["nonverbal_score"])
    ax[0].set_title(TXT["two_axes_fmt"].format(r=r_axes))

    ax[1].plot(nash.alpha, nash.nash_product, color="tab:purple")
    ax[1].axvline(alpha_nash, color="red", ls="--",
                  label=f"{TXT['nash_solution']} \u03b1*={alpha_nash:.3f}")
    ax[1].axvline(0.5, color="gray", ls=":", label=f"{TXT['equal_weights']} \u03b1=0.5")
    ax[1].set_xlabel(TXT["alpha_weight"])
    ax[1].set_ylabel(TXT["nash_product"])
    ax[1].set_title(TXT["nash_curve"])
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_nash_product_and_axes.png", dpi=150)
    plt.close(fig)

    m.to_csv(out / "units_with_scores.csv", index=False)
    print("saved:", out / "units_with_scores.csv")


if __name__ == "__main__":
    main()
