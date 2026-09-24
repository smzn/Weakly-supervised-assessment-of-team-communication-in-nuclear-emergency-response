# Data

## What is here

```
data/
├── transcripts/
│   └── nts_turns_master.csv   one row per utterance, 161 meetings
└── derived/
    └── nonverbal_scores.csv   meeting-level non-verbal scores (written by 04)
```

`nts_turns_master.csv` is the output of the audio pipeline: automatic transcription,
speaker assignment, and acoustic measurements for every utterance. It is the only input
the analysis scripts need.

| Column | Meaning |
|---|---|
| `meeting_id` | meeting identifier, tv001 .. tv161 |
| `start`, `end`, `duration` | utterance boundaries and length, in seconds |
| `speaker` | speaker label from diarization; `UNKNOWN` where it could not be resolved |
| `text` | transcript of the utterance |
| `confidence` | average log probability reported by the recognizer |
| `next_start`, `gap_to_next` | start of the following utterance, and the silence before it |
| `char_len`, `speech_rate_char_per_sec` | characters, and characters per second |
| `rms_mean`, `rms_std` | loudness |
| `zcr_mean`, `spectral_centroid_mean` | zero-crossing rate, spectral centroid |
| `f0_mean`, `f0_std`, `f0_min`, `f0_max`, `f0_range` | pitch statistics |
| `beep_ratio` | share of the utterance covered by the masking beep |
| `meeting_duration_sec`, `silence_ratio` | meeting length, and the share of silence in it |

Utterances whose `speaker` is `UNKNOWN` are excluded in every analysis script.

## What is not here

**Audio and video are not included.** The source material is the video-conference
footage released by the Tokyo Electric Power Company. Obtain it from the publisher and
prepare one 16 kHz mono WAV file per meeting, named `tv001.wav` .. `tv161.wav`, then run:

```bash
python code/02_run_pipeline_batch.py --audio_dir /path/to/wav --outdir data/meeting_output
python code/03_build_turns_master.py --meeting_output data/meeting_output
```

That rewrites `transcripts/nts_turns_master.csv`. Speaker diarization needs a Hugging
Face token in `HF_TOKEN`; without it the pipeline runs but leaves every speaker as
`UNKNOWN`.

Expert survey responses are not included either.

## A note on reproducibility

Speech recognition and diarization are not bit-for-bit reproducible across versions and
machines, so a rebuilt `nts_turns_master.csv` will differ slightly from the one here and
the scores will move a little. Running the analysis on the file in this repository
reproduces the values reported in the paper exactly.

---

# データ

## 置いてあるもの

```
data/
├── transcripts/
│   └── nts_turns_master.csv   発話 1 件につき 1 行、161 会議分
└── derived/
    └── nonverbal_scores.csv   会議単位の非言語スコア（04 が書き出す）
```

`nts_turns_master.csv` は音声パイプラインの出力です。発話ごとの文字起こし、話者、音響
測定値が入っています。分析スクリプトが読むのはこのファイルだけです。列の意味は上の
英語の表を参照してください。`speaker` が `UNKNOWN` の発話は、どの分析でも除いています。

## 置いていないもの

**音声と映像は含みません。** 元になっているのは東京電力が公開したテレビ会議の録画です。
公開元から入手し、16 kHz モノラルの WAV を会議ごとに `tv001.wav`〜`tv161.wav` の名前で
用意してから、次を実行してください。

```bash
python code/02_run_pipeline_batch.py --audio_dir /path/to/wav --outdir data/meeting_output
python code/03_build_turns_master.py --meeting_output data/meeting_output
```

これで `transcripts/nts_turns_master.csv` が作り直されます。話者分離には Hugging Face の
トークンが要ります。`HF_TOKEN` が未設定でもパイプラインは動きますが、話者はすべて
`UNKNOWN` になります。

専門家アンケートの回答も含めていません。

## 再現性について

音声認識と話者分離は、バージョンや実行環境が変わると出力が完全には一致しません。
作り直した `nts_turns_master.csv` はここにあるものとわずかに異なり、スコアも少し動きます。
論文に記載した値をそのまま再現するには、このリポジトリのファイルを使ってください。
