# Weakly supervised assessment of non-technical skills in nuclear emergency teleconferences

Code and transcript data for the study of 161 video-conference sessions recorded at the
Tokyo Electric Power Company during the Fukushima Daiichi nuclear accident.

The study scores each session on communication quality without any human-assigned
labels. Two independent scores are built and then combined:

| Score | Built from | Variable |
|---|---|---|
| Linguistic score | speech acts found in the transcripts, weighted by a Snorkel label model | `y_weak` |
| Nonverbal score | speech rate, pitch variation and loudness | `y_nonverbal` |
| weak COMM score | the two above, weighted by the Nash bargaining solution | `y_combined` |

The linguistic score covers four subcategories of communication: information sharing,
listening, confirmation, and speaking up. Ten indicators are detected morphologically in
the Japanese transcripts, turned into labeling functions, and combined by one label model
per subcategory.

## Repository layout

```
.
├── code/                       every processing and analysis step, as a script
│   ├── 01_audio_pipeline.py    one recording: beep removal, VAD, diarization, ASR, acoustics
│   ├── 02_run_pipeline_batch.py   runs 01 over all recordings
│   ├── 03_build_turns_master.py   merges the per-meeting output into one table
│   ├── 04_nonverbal_scores.py     speech rate, pitch variation, loudness -> y_nonverbal
│   ├── 05_language_score.py       labeling functions and label models -> y_weak
│   ├── 06_combined_score.py       Nash bargaining solution -> y_combined
│   ├── 07_keyword_analysis.py     keyword occurrence against the scores
│   └── nts_common.py              labeling functions, label model, figure labels
├── data/                       transcripts and derived tables; see data/README.md
├── requirements.txt
└── LICENSE
```

Audio and video are not part of this repository. Scripts 01-03 document how the
transcripts were produced; scripts 04-07 reproduce the results of the paper from the
transcripts alone.

## Installation

Python 3.10.

```bash
pip install -r requirements.txt
```

Only the first group of packages in `requirements.txt` is needed to reproduce the
analysis. The audio group (faster-whisper, pyannote.audio, torch and the rest) is needed
only if you rebuild the transcripts from recordings.

## Reproducing the results

```bash
cd code
python 04_nonverbal_scores.py
python 05_language_score.py --lang en
python 06_combined_score.py --lang en
python 07_keyword_analysis.py --lang en
```

Figures and tables are written to `results/`. `--lang ja` produces the same figures with
Japanese labels, which additionally needs `japanize-matplotlib`.

Values reported in the paper, and reproduced by these scripts:

| Quantity | Value |
|---|---|
| Sessions analysed | 161 |
| Correlation between the linguistic and nonverbal scores | r = 0.049 (p = 0.54) |
| Nash weights | alpha = 0.518, beta = 0.482 (exactly 0.500 on standardized axes) |
| Spread of the leave-one-subcategory-out ablation | 0.049 |
| Spearman correlation among the four subcategories | mean 0.118, max 0.247 |
| Keywords, adjusted for the number of utterances | 0 of 75 cells with q < .05; 6 with uncorrected p < .05 |

## Notes on the method

- **Opportunity guard.** Most indicators are absent from most sessions, often because the
  session was too short for the behaviour to have any chance to appear. An absence is
  therefore only treated as evidence when the session had enough utterances; the
  threshold is set so that the absence labeling function covers at most 30% of sessions.
- **One label model per subcategory.** The four subcategories are close to independent in
  this corpus, so a single label model over all labeling functions is pulled towards
  whichever subcategory is internally most consistent. Each subcategory is fitted
  separately and the four scores are averaged with equal weights.
- **Silence is not part of the scores.** The pipeline measures a silence ratio with
  WebRTC VAD, but these meetings were held over a permanently open line, where silence
  cannot be read as a failure to communicate. The value is recorded and left unused.
- **Corpus-internal percentiles for pitch and loudness.** Absolute published thresholds do
  not transfer to this material: pitch depends on the individual speaker, and the
  recordings were not made under uniform conditions. Speech rate, where the literature
  does give usable absolute values, is scored against them directly.

## Citation

Ohba, H., Ito, K., & Mizuno, S. Weakly Supervised Assessment of Non-Technical Skills in
Nuclear Emergency Teleconferences Using Linguistic Features.

## License

MIT. See [LICENSE](LICENSE).

---

# 原子力緊急時テレビ会議における非技術的スキルの弱教師あり評価

福島第一原子力発電所事故の際に東京電力で記録されたテレビ会議 161 件を分析した研究です。
コードと文字起こしデータを置いています。

人手のラベルは使いません。独立した 2 つのスコアを作り、それらを統合して、会議ごとの
コミュニケーションの質をスコア化します。

| スコア | 何から作るか | 変数名 |
|---|---|---|
| 言語スコア | 文字起こしから検出した発話行為を Snorkel の LabelModel で重み付け | `y_weak` |
| 非言語スコア | 発話速度、ピッチの変動、音の大きさ | `y_nonverbal` |
| weak COMM score | 上記 2 つをナッシュ交渉解の重みで統合 | `y_combined` |

言語スコアは、情報共有・傾聴・確認・言い出す力という 4 つの下位要素からなります。
日本語の文字起こしから 10 個の指標を形態素解析で検出し、それぞれをラベリング関数に
変換します。統合は、下位要素ごとに LabelModel を学習させて行います。

## リポジトリの構成

```
.
├── code/                       処理と分析の各段階をスクリプトにしたもの
│   ├── 01_audio_pipeline.py    1 会議分：規制音除去、VAD、話者分離、音声認識、音響特徴
│   ├── 02_run_pipeline_batch.py   全会議に対して 01 を実行
│   ├── 03_build_turns_master.py   会議ごとの出力を 1 つの表にまとめる
│   ├── 04_nonverbal_scores.py     発話速度・ピッチ変動・音量 → y_nonverbal
│   ├── 05_language_score.py       ラベリング関数と LabelModel → y_weak
│   ├── 06_combined_score.py       ナッシュ交渉解 → y_combined
│   ├── 07_keyword_analysis.py     固有名詞の出現とスコアの関係
│   └── nts_common.py              ラベリング関数、LabelModel、図表のラベル
├── data/                       文字起こしと中間データ。詳細は data/README.md
├── requirements.txt
└── LICENSE
```

音声と映像はこのリポジトリに含みません。01〜03 は文字起こしの作り方を示すものです。
論文の結果は、文字起こしがあれば 04〜07 だけで再現できます。

## インストール

Python 3.10 を使います。

```bash
pip install -r requirements.txt
```

分析の再現に必要なのは `requirements.txt` の前半だけです。後半の音声系
（faster-whisper、pyannote.audio、torch など）が要るのは、録画から文字起こしを
作り直すときだけです。

## 結果の再現

```bash
cd code
python 04_nonverbal_scores.py
python 05_language_score.py --lang ja
python 06_combined_score.py --lang ja
python 07_keyword_analysis.py --lang ja
```

図と表は `results/` に出力されます。`--lang ja` は図表のラベルを日本語にする指定で、
`japanize-matplotlib` が必要です。`--lang en` にすると英語のラベルになります。

論文に記載した値のうち、これらのスクリプトで再現できるものは次のとおりです。

| 項目 | 値 |
|---|---|
| 分析対象の会議数 | 161 件 |
| 言語スコアと非言語スコアの相関 | r = 0.049（p = 0.54） |
| ナッシュ交渉解の重み | α = 0.518、β = 0.482（標準化すると厳密に 0.500） |
| 下位要素を 1 つずつ除いたアブレーションの幅 | 0.049 |
| 4 下位要素間の Spearman 相関 | 平均 0.118、最大 0.247 |
| 発話数を揃えた固有名詞の検定 | q < .05 は 75 セル中 0、補正前 p < .05 は 6 |

## 手法上の注意点

- **機会ガード**：多くの指標は、ほとんどの会議で出現しません。会議が短く、その行動が
  現れる機会自体がなかった場合が多いためです。そこで「出現しなかったこと」を根拠と
  するのは、発話数が一定以上あった会議に限りました。閾値は、不在のラベリング関数が
  判定する会議が全体の 30% を超えないように決めています。
- **下位要素ごとの LabelModel**：本コーパスでは 4 つの下位要素がほぼ独立しているため、
  すべてのラベリング関数を 1 つの LabelModel に入れると、内部一貫性が最も高い要素に
  スコアが引きずられます。下位要素ごとに学習させ、4 つのスコアを等重みで平均します。
- **沈黙はスコアに含めない**：パイプラインは WebRTC VAD で沈黙比率を算出しますが、
  常時接続で行われる会議では、沈黙をそのままコミュニケーションの不全とは読めません。
  値は記録するだけで、スコアには使っていません。
- **ピッチと音量はコーパス内のパーセンタイル**：ピッチは話者ごとの差が大きく、録音条件も
  揃っていません。文献の絶対値はそのまま持ち込めないと判断しました。絶対値が使える
  発話速度についてのみ、文献の基準値を閾値としています。

## 引用

Ohba, H., Ito, K., & Mizuno, S. Weakly Supervised Assessment of Non-Technical Skills in
Nuclear Emergency Teleconferences Using Linguistic Features.

## ライセンス

MIT ライセンスです。[LICENSE](LICENSE) を参照してください。
