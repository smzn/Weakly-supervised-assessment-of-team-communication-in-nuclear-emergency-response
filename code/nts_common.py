"""Shared pieces of the analysis: labeling functions, label model, figure labels.

The labeling functions match Japanese morphemes and expressions, so their patterns stay
in Japanese. Everything that is displayed (figure labels, table headers, printed text)
goes through the LABELS dictionary and can be switched between English and Japanese
with the --lang option of each script.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from janome.tokenizer import Tokenizer
from snorkel.labeling.model import LabelModel

SEED = 42
ABSTAIN, HIGH, OTHER = -1, 1, 0

# Coverage target of the "absence" labeling function: at most this share of meetings may
# receive a negative vote from it. See the opportunity guard in guard_for().
ABSENCE_TARGET_COV = 0.30

REPO = Path(__file__).resolve().parent.parent
DEFAULT_TURNS = REPO / "data" / "transcripts" / "nts_turns_master.csv"
DEFAULT_NONVERBAL = REPO / "data" / "derived" / "nonverbal_scores.csv"
DEFAULT_OUT = REPO / "results"

# ---------------------------------------------------------------------------
# Morpheme-based labeling functions (Japanese patterns; do not translate)
# ---------------------------------------------------------------------------
_T = Tokenizer()


def parse(text):
    return [(x.surface, *x.part_of_speech.split(",")[:2], x.base_form)
            for x in _T.tokenize(text)]


# --- Confirmation: split closed-loop into request (call-out) and report (check-back) ---
CONF_NOUN = {"確認", "復唱", "再確認", "整理"}
RE_REQUEST = re.compile(r"お願い|ください|下さい|くれ|もらえ|もらい|いただけ|"
                        r"ほしい|欲しい|しといて|しておいて")
RE_QUESTION = re.compile(r"ですか|でしょうか|ますか|いいです|よろしい|[かの]$|^です")


def conf_features(text):
    toks = parse(text)
    surf = [t[0] for t in toks]
    res = dict(conf_request=0, conf_report=0, conf_ongoing=0)
    for i, (s, p_big, p_sub, base) in enumerate(toks):
        if s not in CONF_NOUN and base not in CONF_NOUN:
            continue
        tail = "".join(surf[i + 1:i + 7])
        if RE_REQUEST.search(tail):
            res["conf_request"] = 1
        elif "中" in surf[i + 1:i + 3]:
            res["conf_ongoing"] = 1
        elif RE_QUESTION.search(tail):
            res["conf_request"] = 1
        else:
            res["conf_report"] = 1
    return res


# --- Speaking up: direct vs. mitigated, following Linde (1988) ---
RE_PROPOSAL = re.compile(r"(た|し)方が(いい|良|よ)|べきだ|べきです|べきじゃ|"
                         r"ましょう|ませんか|どうでしょう|どうですか|提案|してはどう")
RE_HEDGED = re.compile(r"じゃないです?か|ではないでしょうか|のでは(ない|ありません)|"
                       r"(ない|ん)です?か[ねな]|ちょっと(待|厳しい|難しい|きつい)")
RE_RISK = re.compile(r"危[険な]|リスク|懸念|心配|まずい|恐れ|支障|不安|"
                     r"間に合わ|できません|できない|無理|問題")


def speakup_features(text):
    s = "".join(t[0] for t in parse(text))
    return dict(su_proposal=int(bool(RE_PROPOSAL.search(s))),
                su_hedged=int(bool(RE_HEDGED.search(s))),
                su_risk=int(bool(RE_RISK.search(s))))


# --- Listening: exclude filler tokens ---
RE_AIZUCHI = re.compile(r"^(はい|ええ|うん|そう|そうです|そうですね|なるほど|"
                        r"了解|承知|分かりました|わかりました|オーケー|オッケー)[はねよ。、！]*$")
RE_UNDERSTAND = re.compile(r"了解|承知|分かりました|わかりました|なるほど")


def listen_features(text):
    s = text.strip()
    is_filler = bool(re.fullmatch(r"[えあうんーっ、。 　]*", s))
    return dict(li_aizuchi=int(bool(RE_AIZUCHI.match(s)) and not is_filler),
                li_understand=int(bool(RE_UNDERSTAND.search(s))))


# --- Information sharing ---
RE_ENUM = re.compile(r"まず(?!い)|次に|それから|最後に|一つ目|二つ目|"
                     r"[一二三四1234]点目|[一二三]つ目|順番に|続いて")
RE_CLOSING = re.compile(r"以上|以上です|以上で|終わります|終わりです")


def disc_features(text):
    toks = parse(text)
    s = "".join(t[0] for t in toks)
    return dict(dm_enum=int(bool(RE_ENUM.search(s))),
                dm_closing=int(bool(RE_CLOSING.search(s))),
                dm_conn=int(any(p == "接続詞" for _, p, _, _ in toks)))


FEATURE_FUNCS = [conf_features, speakup_features, listen_features, disc_features]

CATEGORY = {
    "info":    ["dm_enum", "dm_closing", "dm_conn"],
    "listen":  ["li_aizuchi", "li_understand"],
    "conf":    ["conf_request", "conf_report"],
    "speakup": ["su_proposal", "su_hedged", "su_risk"],
}
RATIO_COLS = [c for cols in CATEGORY.values() for c in cols]

# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------
LABELS = {
    "en": {
        "category": {"info": "Information sharing", "listen": "Listening",
                     "conf": "Confirmation", "speakup": "Speaking up"},
        "indicator": {"dm_enum": "Enumeration", "dm_closing": "Closing",
                      "dm_conn": "Conjunction", "li_aizuchi": "Backchannel",
                      "li_understand": "Understanding",
                      "conf_request": "Confirmation request",
                      "conf_report": "Confirmation report",
                      "su_proposal": "Direct proposal", "su_hedged": "Hedged assertion",
                      "su_risk": "Risk mention"},
        "keyword": {"E_官邸": "E_PM Office", "E_吉田所長": "E_Supt. Yoshida",
                    "E_東電本店": "E_TEPCO HQ", "I_1号機": "I_Unit 1", "I_2号機": "I_Unit 2",
                    "I_3号機": "I_Unit 3", "I_4号機": "I_Unit 4", "O_ベント": "O_Venting",
                    "O_ポンプ": "O_Pump", "O_注水": "O_Water injection", "O_燃料": "O_Fuel",
                    "O_圧力": "O_Pressure", "O_炉心": "O_Reactor core",
                    "O_燃料棒": "O_Fuel rod", "O_サプチャン": "O_Suppression chamber",
                    "O_ドライウェル": "O_Drywell"},
        "text": {
            "language_score": "Linguistic score",
            "nonverbal_score": "Nonverbal score",
            "comm_score": "weak COMM score",
            "n_meetings": "Number of sessions",
            "subcategory": "Subcategory",
            "indicator": "Indicator",
            "zero_rate": "Zero rate",
            "guard": "Opportunity guard",
            "guard_none": "none",
            "utterances": "utterances",
            "absence_cov": "Absence-LF coverage",
            "excluded": "Excluded subcategory",
            "pearson": "Pearson r",
            "keyword": "Keyword",
            "n_present": "Meetings with keyword",
            "test": "Test",
            "mean_diff": "Mean diff",
            "q_bh": "q (BH)",
            "significant": "Significant (q<.05)",
            "n_too_small": "n too small",
            "present": "Present",
            "absent": "Absent",
            "outcome": "Outcome",
            "coef": "Coef",
            "q_label": "BH-adjusted q",
            "other_meetings": "Other sessions",
            "top20": "Top 20% linguistic score",
            "scatter_title_fmt": "Linguistic score vs. Nonverbal score (r = {r:.3f})",
            "two_axes_fmt": "Relation between the two axes  r = {r:.3f}",
            "nash_curve": "Nash product as a function of \u03b1",
            "nash_solution": "Nash solution",
            "equal_weights": "Equal weights",
            "alpha_weight": "\u03b1 (weight on the linguistic score)",
            "nash_product": "Nash product N(\u03b1)",
        },
    },
    "ja": {
        "category": {"info": "情報共有", "listen": "傾聴",
                     "conf": "確認", "speakup": "言い出す力"},
        "indicator": {"dm_enum": "列挙", "dm_closing": "締めくくり", "dm_conn": "接続詞",
                      "li_aizuchi": "相づち", "li_understand": "理解表明",
                      "conf_request": "確認要求", "conf_report": "確認報告",
                      "su_proposal": "直接提案", "su_hedged": "留保つき主張",
                      "su_risk": "リスク言及"},
        "keyword": {k: k for k in ["E_官邸", "E_吉田所長", "E_東電本店", "I_1号機", "I_2号機",
                                   "I_3号機", "I_4号機", "O_ベント", "O_ポンプ", "O_注水",
                                   "O_燃料", "O_圧力", "O_炉心", "O_燃料棒", "O_サプチャン",
                                   "O_ドライウェル"]},
        "text": {
            "language_score": "言語スコア",
            "nonverbal_score": "非言語スコア",
            "comm_score": "weak COMM score",
            "n_meetings": "会議数",
            "subcategory": "下位要素",
            "indicator": "指標",
            "zero_rate": "ゼロ率",
            "guard": "機会ガード",
            "guard_none": "設定なし",
            "utterances": "発話",
            "absence_cov": "不在LFの判定率",
            "excluded": "除外したカテゴリ",
            "pearson": "Pearson",
            "keyword": "固有名詞",
            "n_present": "出現会議数",
            "test": "検定",
            "mean_diff": "平均差",
            "q_bh": "p値(BH補正)",
            "significant": "有意(BH<.05)",
            "n_too_small": "n不足",
            "present": "含む方",
            "absent": "含まない方",
            "outcome": "目的変数",
            "coef": "係数",
            "q_label": "BH 補正後 q 値",
            "other_meetings": "その他",
            "top20": "言語スコア上位 20%",
            "scatter_title_fmt": "言語スコアと非言語スコアの関係（r = {r:.3f}）",
            "two_axes_fmt": "2 軸の関係  r = {r:.3f}",
            "nash_curve": "ナッシュ積の推移",
            "nash_solution": "ナッシュ解",
            "equal_weights": "等重み",
            "alpha_weight": "α（言語スコアの重み）",
            "nash_product": "ナッシュ積 N(α)",
        },
    },
}


def get_labels(lang):
    """Return the label set for the requested language and set up matplotlib for it."""
    if lang not in LABELS:
        raise ValueError(f"unknown language: {lang}")
    if lang == "ja":
        # Japanese figure labels need a font that carries the glyphs.
        import japanize_matplotlib  # noqa: F401
    return LABELS[lang]


def add_common_args(ap):
    ap.add_argument("--turns", default=str(DEFAULT_TURNS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--lang", default="en", choices=["en", "ja"],
                    help="language of figure labels and table headers")
    return ap


# ---------------------------------------------------------------------------
# Building the meeting-level table and the language score
# ---------------------------------------------------------------------------
def load_turns(path):
    """Utterance table, with unresolved speakers dropped as in the study."""
    df = pd.read_csv(path)
    df["text"] = df.text.fillna("").astype(str)
    df = df[df.speaker != "UNKNOWN"].copy()
    df["dur"] = df.duration
    return df


def build_units(df, progress=True):
    """Add the indicator features to every utterance and aggregate per meeting.

    Each indicator becomes a time ratio: the total duration of the utterances in which
    it fires, divided by the total speaking time of the meeting.
    """
    if progress:
        from tqdm import tqdm
        tqdm.pandas(desc="Tokenizing")
        mapper = df.text.progress_map
    else:
        mapper = df.text.map

    feats = [pd.DataFrame(list(mapper(fn)), index=df.index) for fn in FEATURE_FUNCS]
    df = pd.concat([df] + feats, axis=1)

    for c in RATIO_COLS:
        df["dur_" + c] = df.dur * df[c]

    units = (df.groupby("meeting_id")
               .agg(tot=("dur", "sum"), n=("text", "size"),
                    **{c: ("dur_" + c, "sum") for c in RATIO_COLS})
               .reset_index())
    for c in RATIO_COLS:
        units[c] = units[c] / units.tot.clip(lower=1e-6)
    return df, units


def guard_for(v, n, target=ABSENCE_TARGET_COV):
    """Utterance count above which a zero time ratio counts as evidence of absence.

    Many indicators are zero in most meetings, often simply because the meeting was too
    short for the behaviour to have a chance to appear. The guard keeps the negative
    vote for the meetings that had the most opportunity, capping its coverage at
    `target`. Returns -inf when the indicator is non-zero often enough that no guard is
    needed.
    """
    z = (v == 0)
    if z.mean() <= target:
        return -np.inf
    return float(np.quantile(n[z], 1 - target / z.mean()))


def category_L(u, cols):
    """Label matrix for one subcategory: three LFs per indicator."""
    L, names = [], []
    for c in cols:
        v = u[c].values
        nz = v[v > 0]
        med = np.median(nz) if len(nz) else np.inf
        g = guard_for(v, u.n.values)
        L += [np.where(v > 0, HIGH, ABSTAIN),          # occurrence
              np.where(v > med, HIGH, ABSTAIN),        # amount above the non-zero median
              np.where((v == 0) & (u.n.values >= g), OTHER, ABSTAIN)]  # guarded absence
        names += [f"{c}:occur", f"{c}:amount", f"{c}:absent"]
    return np.array(L).T, names


def fit_lm(L):
    """Fit a LabelModel on the columns that vote at least once."""
    alive = [j for j in range(L.shape[1]) if (L[:, j] != ABSTAIN).any()]
    m = LabelModel(cardinality=2, verbose=False)
    m.fit(L[:, alive], n_epochs=500, log_freq=500, seed=SEED)
    return m.predict_proba(L[:, alive])[:, 1], L[:, alive]


def language_score(units):
    """One LabelModel per subcategory, averaged with equal weights into y_weak.

    The four subcategories are close to independent in this corpus, so a single
    LabelModel over all labeling functions would be dominated by whichever subcategory
    is internally most consistent.
    """
    cat_scores, cat_L = {}, {}
    for cat, cols in CATEGORY.items():
        Lc, _ = category_L(units, cols)
        y, alive = fit_lm(Lc)
        cat_scores[cat] = y
        cat_L[cat] = alive
    cs = pd.DataFrame(cat_scores, index=units.index)
    units = pd.concat([units, cs], axis=1)
    units["y_weak"] = cs.mean(axis=1)
    return units, cs, cat_L


def benjamini_hochberg(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, idx in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[idx] * n / r)
        adj[idx] = prev
    return adj
