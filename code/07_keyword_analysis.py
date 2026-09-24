"""Keyword occurrence and the weak COMM score.

Keywords are named entities of the accident (organisations, reactor units, operations).
They are extracted from the Japanese transcripts with a normalisation map that folds
spelling variants and recognition errors onto one label, then displayed in the language
chosen with --lang.

Two analyses:
  1. Per keyword, meetings that contain it vs. meetings that do not (Shapiro-Wilk first,
     then Welch's t or Mann-Whitney), with Benjamini-Hochberg correction.
  2. Per subcategory, an OLS regression of the score on keyword presence with log(number
     of utterances) as a covariate, so that meeting length cannot drive the association.

Outputs (under --out):
    keyword_tests_weakCOMM.csv
    keyword_regression_adjusted.csv
    fig_keyword_boxplots.png
    fig_keyword_adjusted_heatmap.png

Example:
    python 07_keyword_analysis.py --lang en
"""

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import nts_common as nts

warnings.filterwarnings("ignore")

# Normalisation map: Japanese spelling variants and ASR errors -> one keyword label.
# E_ entities, I_ reactor units, O_ operations and plant items.
SYNONYM_RULES = {
    r'官邸': 'E_官邸',
    r'吉田|所長|室長': 'E_吉田所長',
    r'本店|東電': 'E_東電本店',
    r'1号(機)?|１号(機)?|一号(機)?': 'I_1号機',
    r'2号(機)?|２号(機)?|二号(機)?': 'I_2号機',
    r'3号(機)?|３号(機)?|三号(機)?': 'I_3号機',
    r'4号(機)?|４号(機)?|四号(機)?': 'I_4号機',
    r'弁当|ベント|弁と': 'O_ベント',
    r'ポンプ': 'O_ポンプ',
    r'注水': 'O_注水',
    r'燃料': 'O_燃料',
    r'圧力': 'O_圧力',
    r'路芯|炉心|露進': 'O_炉心',
    r'燃料棒': 'O_燃料棒',
    r'サプチャン|サプレッション・チェンバー|サプレッション': 'O_サプチャン',
    r'ドライウェル|ドライベル': 'O_ドライウェル',
}

# Reactor units are handled by MACHINE_PATTERN in stage 1, so they are excluded from
# stage 2 to avoid counting them twice.
ITEM_KEYS = {
    r'1号(機)?|１号(機)?|一号(機)?',
    r'2号(機)?|２号(機)?|二号(機)?',
    r'3号(機)?|３号(機)?|三号(機)?',
    r'4号(機)?|４号(機)?|四号(機)?',
}
MACHINE_PATTERN = re.compile(r'([1234１２３４一二三四][号機・、, とと、~〜\-ー]*)+号(機)?')
NUM_MAP = {'1': '1', '１': '1', '一': '1', '2': '2', '２': '2', '二': '2',
           '3': '3', '３': '3', '三': '3', '4': '4', '４': '4', '四': '4'}


def extract_keywords(df_text, kw_labels):
    """Return a meeting x keyword count table."""
    rules_stage2 = {k: v for k, v in SYNONYM_RULES.items() if k not in ITEM_KEYS}
    compiled_stage2 = [(re.compile(p), r) for p, r in rules_stage2.items()]
    combined_pattern = re.compile('|'.join(f'({p})' for p in rules_stage2))

    records = []
    for _, row in df_text.iterrows():
        m_id, text = row['meeting_id'], row['text']
        if not text or pd.isna(text):
            continue

        # Stage 1: reactor units, including forms such as "1、2号機" that name several.
        for match_obj in MACHINE_PATTERN.finditer(text):
            for num in set(re.findall(r'[1234１２３４一二三四]', match_obj.group(0))):
                records.append((m_id, f"I_{NUM_MAP[num]}号機"))

        # Stage 2: everything else.
        for match in combined_pattern.findall(text):
            matched_word = next(m for m in match if m)
            for pattern, resolved_name in compiled_stage2:
                if pattern.match(matched_word):
                    records.append((m_id, resolved_name))
                    break

    pairs = pd.DataFrame(records, columns=['meeting_id', 'keyword'])
    pairs['keyword'] = pairs['keyword'].map(kw_labels)
    assert pairs['keyword'].notna().all(), "a keyword is missing from the label map"

    matrix = pd.crosstab(pairs['meeting_id'], pairs['keyword'])
    matrix.columns.name = None
    return matrix.reset_index()


def run_tests(dm, wcols, target, TXT):
    """Presence vs. absence of each keyword, with BH correction."""
    rows = []
    for w in wcols:
        gp = dm.loc[dm[w] > 0, target].values
        ga = dm.loc[dm[w] == 0, target].values
        if len(gp) < 3 or len(ga) < 3:
            rows.append({TXT["keyword"]: w, TXT["n_present"]: len(gp),
                         TXT["test"]: TXT["n_too_small"],
                         TXT["mean_diff"]: np.nan, "p": np.nan})
            continue
        normal = stats.shapiro(gp)[1] > .05 and stats.shapiro(ga)[1] > .05
        if normal:
            _, pv = stats.ttest_ind(gp, ga, equal_var=False)
            meth = "Welch t"
        else:
            _, pv = stats.mannwhitneyu(gp, ga, alternative="two-sided")
            meth = "Mann-Whitney"
        rows.append({TXT["keyword"]: w, TXT["n_present"]: len(gp), TXT["test"]: meth,
                     TXT["mean_diff"]: gp.mean() - ga.mean(), "p": pv})

    r = pd.DataFrame(rows)
    ok = r["p"].notna()
    r[TXT["q_bh"]] = np.nan
    r.loc[ok, TXT["q_bh"]] = nts.benjamini_hochberg(r.loc[ok, "p"].values)
    r[TXT["significant"]] = r[TXT["q_bh"]] < .05
    return r.sort_values(TXT["q_bh"]).reset_index(drop=True)


def main():
    ap = nts.add_common_args(argparse.ArgumentParser())
    ap.add_argument("--scores", default=None,
                    help="units_with_scores.csv from 06_combined_score.py")
    args = ap.parse_args()
    L = nts.get_labels(args.lang)
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use("ggplot")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    TXT, CAT = L["text"], L["category"]

    scores_path = Path(args.scores) if args.scores else out / "units_with_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"{scores_path} not found; run 06_combined_score.py first.")
    m = pd.read_csv(scores_path)

    df = nts.load_turns(args.turns)
    df_text = df.groupby("meeting_id")["text"].agg("\n".join).reset_index()
    df_matrix = extract_keywords(df_text, L["keyword"])
    print(f"{df_matrix.shape[1]-1} keywords / {len(df_matrix)} meetings with at least one keyword")
    print(df_matrix.set_index("meeting_id").sum().sort_values(ascending=False).to_string())

    dm = df_text[["meeting_id"]].merge(df_matrix, on="meeting_id", how="left").fillna(0)
    wcols = [c for c in dm.columns if c != "meeting_id"]
    dm[wcols] = dm[wcols].astype(int)
    dm = dm.merge(m[["meeting_id", "y_weak", "y_nonverbal", "y_combined"]],
                  on="meeting_id", how="inner")
    print(f"\nAnalyzed: {len(dm)} meetings / {len(wcols)} keywords")

    # --- 1. presence vs. absence, with the weak COMM score as the outcome ---
    res = run_tests(dm, wcols, "y_combined", TXT)
    res["Cohen_d"] = res[TXT["mean_diff"]] / m.y_combined.std()
    print("\n" + res[[TXT["keyword"], TXT["n_present"], TXT["test"], TXT["mean_diff"],
                      "Cohen_d", "p", TXT["q_bh"]]].round(4).to_string(index=False))
    print(f"\nSignificant after BH correction: {int(res[TXT['significant']].sum())} keywords")
    print(f"|d| mean {res.Cohen_d.abs().mean():.3f}  max {res.Cohen_d.abs().max():.3f}  "
          f"|d|>0.2: {int((res.Cohen_d.abs() > 0.2).sum())} keywords")
    res.to_csv(out / "keyword_tests_weakCOMM.csv", index=False)

    # --- box plots, one panel per keyword ---
    res_i = res.set_index(TXT["keyword"])
    words_sorted = sorted(wcols)
    ncol = 5
    nrow = int(np.ceil(len(words_sorted) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow))
    axes = np.atleast_2d(axes)
    # Fixed layout in every panel: absent on the left, present on the right.
    pal = {TXT["absent"]: "#8fa8bd", TXT["present"]: "#e8845c"}
    for ax, w in zip(axes.ravel(), words_sorted):
        d = dm.assign(**{TXT["keyword"]: np.where(dm[w] > 0, TXT["present"], TXT["absent"])})
        r = res_i.loc[w]
        sns.boxplot(data=d, x=TXT["keyword"], y="y_combined", ax=ax,
                    order=[TXT["absent"], TXT["present"]], palette=pal)
        sig = r[TXT["q_bh"]] < .05
        ax.set_title(f"{w}\np={r['p']:.4f}, q={r[TXT['q_bh']]:.4f}",
                     fontsize=11, color="red" if sig else "black")
        ax.set_xlabel("")
        ax.set_ylabel(TXT["comm_score"])
        ax.set_ylim(-0.02, 1.02)
    for ax in axes.ravel()[len(words_sorted):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "fig_keyword_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Uncorrected p < .05: {int((res_i['p'] < .05).sum())} keywords  "
          f"{list(res_i[res_i['p'] < .05].index)}")

    # --- 2. regression adjusted for the number of utterances ---
    dmr = dm.merge(m[["meeting_id", "n"] + list(nts.CATEGORY)], on="meeting_id", how="left")
    dmr["logn"] = np.log(dmr.n)
    targets = list(nts.CATEGORY) + ["y_combined"]
    tlab = {**CAT, "y_combined": TXT["comm_score"]}
    qmat, rows = {}, []
    for tgt in targets:
        ps = []
        for w in wcols:
            X = sm.add_constant(pd.DataFrame({"kw": (dmr[w] > 0).astype(int),
                                              "logn": dmr.logn}))
            f = sm.OLS(dmr[tgt], X).fit()
            ps.append(f.pvalues["kw"])
            rows.append({TXT["outcome"]: tlab[tgt], TXT["keyword"]: w,
                         TXT["coef"]: f.params["kw"], "p": f.pvalues["kw"]})
        qmat[tlab[tgt]] = nts.benjamini_hochberg(ps)
    q = pd.DataFrame(qmat, index=wcols)

    fig, ax = plt.subplots(figsize=(6.8, 6))
    sns.heatmap(q, ax=ax, vmin=0, vmax=1, cmap="Blues_r", annot=True, fmt=".2f",
                annot_kws={"size": 8}, cbar_kws={"label": TXT["q_label"]},
                linewidths=.5, linecolor="#cccccc")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    plt.setp(ax.get_xticklabels(), ha="right")
    fig.tight_layout()
    fig.savefig(out / "fig_keyword_adjusted_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nCells with q < 0.05: {int((q < .05).sum().sum())} / {q.size}   "
          f"min q = {q.min().min():.3f}")
    reg = pd.DataFrame(rows)
    weak = reg[reg.p < .05].sort_values("p")
    print(f"Cells with uncorrected p < 0.05: {len(weak)} / {len(reg)}")
    print(weak.round(4).to_string(index=False))
    print(f"|Coef|: mean {reg[TXT['coef']].abs().mean():.3f}  "
          f"max {reg[TXT['coef']].abs().max():.3f}")
    reg.to_csv(out / "keyword_regression_adjusted.csv", index=False)
    print("saved:", out / "keyword_regression_adjusted.csv")


if __name__ == "__main__":
    main()
