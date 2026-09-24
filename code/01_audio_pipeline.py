# meeting_audio_pipeline_v2.py
#
# [Fix 15] Check whether the "beep" tones used to mask personal information
# contaminate (a) the acoustic features (RMS, F0, ZCR, spectral centroid),
# (b) speaker diarization, and (c) the ASR transcript and speech rate, and
# remove their influence.
#
# Approach:
#   - Measuring a sample of the beep shows that it is a
#     pure tone confined to an extremely narrow band around 1000 Hz with very
#     low spectral flatness (in-band energy ratio > 0.99, spectral flatness
#     < 0.003). We use these properties to decide, frame by frame, how
#     beep-like a frame is.
#
#   - [Fix 15-a] Acoustic features: beeps are detected per STFT frame, and the
#     samples belonging to beep frames are dropped immediately before the
#     acoustic features (RMS, ZCR, spectral centroid, F0) are computed, so the
#     features come from "clean" audio only.
#
#   - [Fix 15-b] Diarization and ASR (effect on speech rate): fixing the
#     acoustic features alone turned out not to be enough. ASR and diarization
#     run over the whole original recording, beeps included, which left three
#     problems: (1) Whisper may transcribe the beep stretches as if they were
#     speech; (2) the duration of a turn (= end - start) includes the beep
#     time while char_len does not include any characters for it, so
#     speech_rate_char_per_sec (= char_len / duration) comes out unreasonably
#     low; (3) a 1000 Hz pure tone can confuse the speaker embeddings and throw
#     off the separation of speakers. We therefore build a copy of the audio in
#     which the beep samples are replaced by silence (0) — the timeline and the
#     sample count are identical to the original, so start/end still refer to
#     times in the original audio file — and run both diarization and ASR on
#     this "beep-silenced" audio. VAD then treats the beep stretches as
#     silence, which keeps the beeps out of each turn's transcript, duration
#     and speech_rate_char_per_sec. A short fade (10 ms) is applied around the
#     replaced samples to avoid clicks at the boundaries.
#
#   - Acoustic feature extraction ([Fix 15-a]) still works on the original,
#     un-silenced audio and applies the beep mask as an exclusion. The
#     silenced audio would give mathematically identical results, but this way
#     we avoid carrying a second copy of the waveform around.
#
#   - Every turn now also reports `beep_ratio`, the fraction of its duration
#     that was judged to be a beep, so that later analyses can identify and
#     drop turns that are heavily contaminated.
#
#   - When too little clean audio is left (under 0.2 s), the acoustic features
#     are reported as missing (NaN), as in the original code. beep_ratio is
#     always recorded.

import os
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import webrtcvad
import parselmouth
import torch

from pydub import AudioSegment
from faster_whisper import WhisperModel
#from pyannote.audio import Pipeline
from pyannote.core import Segment

SR = 16000

# --- [Fix 15] Beep detection parameters (taken from measurements of a beep sample) ---
BEEP_TARGET_FREQ = 1000.0     # Center frequency of the beep (Hz); measured peak is 1000.1 Hz.
BEEP_HALF_WIDTH = 75.0        # Tolerance around the center frequency (Hz), to absorb slight tone differences.
BEEP_BAND_RATIO_THRESH = 0.5  # In-band energy ratio above this marks a frame as a beep candidate.
BEEP_FLATNESS_THRESH = 0.01   # Spectral flatness below this means tone-like, i.e. a beep candidate.
BEEP_FRAME_LENGTH = 2048
BEEP_HOP_LENGTH = 512
BEEP_MIN_CLEAN_SEC = 0.2      # If less audio than this survives the exclusion, report the features as NaN.

@dataclass
class SegmentRow:
    start: float
    end: float
    duration: float
    speaker: str
    text: str = ""
    confidence: float = np.nan

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_audio_mono(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y

def save_wav(y, path, sr=SR):
    sf.write(path, y, sr)

def frame_generator(audio, sample_rate, frame_duration_ms=30):
    n = int(sample_rate * (frame_duration_ms / 1000.0))
    offset = 0
    while offset + n <= len(audio):
        yield audio[offset:offset+n]
        offset += n

def pcm16_bytes(frame):
    frame = np.clip(frame, -1.0, 1.0)
    pcm = (frame * 32767).astype(np.int16)
    return pcm.tobytes()

def detect_speech_regions_vad(y, sr=SR, aggressiveness=2, frame_ms=30, min_region_ms=300):
    # [Fix 15] Unchanged: this decides speech vs. silence, and a beep is still a
    # sound, so it stays inside the speech regions rather than being mistaken
    # for silence.
    vad = webrtcvad.Vad(aggressiveness)
    flags = []
    times = []
    hop = int(sr * frame_ms / 1000)
    for i, frame in enumerate(frame_generator(y, sr, frame_ms)):
        is_speech = vad.is_speech(pcm16_bytes(frame), sr)
        flags.append(is_speech)
        times.append(i * frame_ms / 1000.0)

    regions = []
    in_seg = False
    s = None
    for i, flag in enumerate(flags):
        t = times[i]
        if flag and not in_seg:
            s = t
            in_seg = True
        elif not flag and in_seg:
            e = t + frame_ms / 1000.0
            if (e - s) * 1000 >= min_region_ms:
                regions.append((s, e))
            in_seg = False
    if in_seg:
        e = times[-1] + frame_ms / 1000.0
        if (e - s) * 1000 >= min_region_ms:
            regions.append((s, e))
    return pd.DataFrame(regions, columns=["start", "end"]).assign(duration=lambda d: d.end - d.start)

def complement_silence_regions(total_dur, speech_df):
    if speech_df.empty:
        return pd.DataFrame([{"start": 0.0, "end": total_dur, "duration": total_dur}])
    rows = []
    prev = 0.0
    for _, r in speech_df.sort_values("start").iterrows():
        if r["start"] > prev:
            rows.append({"start": prev, "end": r["start"], "duration": r["start"] - prev})
        prev = max(prev, r["end"])
    if prev < total_dur:
        rows.append({"start": prev, "end": total_dur, "duration": total_dur - prev})
    return pd.DataFrame(rows)

def diarize_audio(audio_path, hf_token, waveform=None, sr=SR):
    """[Fix 15-b] If a waveform (a 1-D numpy array) is given, diarize that
    instead, which is how the beep-silenced audio gets in. When waveform is
    None the original audio_path is used as before (backward compatible).
    """
    try:
        from pyannote.audio import Pipeline
    except Exception as e:
        print(f"[WARN] pyannote.audio import failed: {e}")
        return pd.DataFrame(columns=["start","end","duration","speaker"])

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        if waveform is not None:
            audio_input = {
                "waveform": torch.from_numpy(waveform).float().unsqueeze(0),  # (1, n_samples)
                "sample_rate": sr,
            }
        else:
            audio_input = audio_path
        diarization = pipeline(audio_input)
        rows = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            rows.append({
                "start": float(turn.start),
                "end": float(turn.end),
                "duration": float(turn.end - turn.start),
                "speaker": speaker,
            })
        return pd.DataFrame(rows).sort_values(["start","end"]).reset_index(drop=True)
    except Exception as e:
        print(f"[WARN] diarization pipeline failed: {e}")
        return pd.DataFrame(columns=["start","end","duration","speaker"])

def transcribe_audio(audio_path, model_size="large-v3", language="ja", waveform=None, total_dur=None):
    """[Fix 15-b] If a waveform (a 1-D float32 numpy array at 16 kHz) is given,
    transcribe that instead, which is how the beep-silenced audio gets in;
    faster-whisper accepts a numpy array directly as its audio argument. When
    waveform is None the original audio_path is used as before (backward
    compatible).

    [Fix 16] Guarding against hallucinations:
      - The default model_size was changed from "small" to "large-v3". The
        batch driver used in practice, 02_run_pipeline_batch.py, already passed
        large-v3 explicitly, but running this script on its own with minimal
        arguments fell back to small, which produced far more looping
        hallucinations (the same short phrase repeated endlessly).
      - condition_on_previous_text=False: not conditioning on the previous
        output stops a hallucination, once started, from running away into an
        endless repetition.
      - hallucination_silence_threshold=2.0: faster-whisper's built-in
        hallucination detection, which skips a stretch that looks silent but
        keeps producing unusual output.
      - Even with both of those, isolated stock hallucination phrases such as
        "ご視聴ありがとうございました" ("thank you for watching") still appeared
        over silence and at the tail of short recordings, so a safety net was
        added: a segment whose end lies beyond the actual audio length
        total_dur is physically impossible and is therefore dropped as a
        hallucination.
    """
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    audio_input = waveform.astype(np.float32) if waveform is not None else audio_path
    segs, info = model.transcribe(
        audio_input,
        language=language,
        vad_filter=True,
        condition_on_previous_text=False,
        hallucination_silence_threshold=2.0,
    )
    rows = []
    n_dropped_oob = 0
    for s in segs:
        # [Fix 16] Treat segments timestamped past the end of the actual audio
        # as hallucinations and drop them (with a 0.5 s tolerance).
        if total_dur is not None and float(s.start) > total_dur + 0.5:
            n_dropped_oob += 1
            continue
        seg_start = float(s.start)
        seg_end = float(s.end) if total_dur is None else min(float(s.end), total_dur)
        rows.append({
            "start": seg_start,
            "end": seg_end,
            "duration": seg_end - seg_start,
            "text": s.text.strip(),
            "confidence": float(np.mean(s.avg_logprob if hasattr(s, "avg_logprob") else np.nan))
                if hasattr(s, "avg_logprob") else np.nan
        })
    if n_dropped_oob:
        print(f"[WARN] Dropped {n_dropped_oob} segment(s) as hallucinations: "
              f"their timestamps ran past the end of the audio.")
    return pd.DataFrame(rows).sort_values(["start", "end"]).reset_index(drop=True)

def overlap(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def assign_text_to_speakers(diar_df, asr_df, min_overlap=0.1):
    # [Fix 15] Unchanged.
    rows = []
    for _, a in asr_df.iterrows():
        best_idx = None
        best_ov = 0.0
        for idx, d in diar_df.iterrows():
            ov = overlap(a.start, a.end, d.start, d.end)
            if ov > best_ov:
                best_ov = ov
                best_idx = idx
        speaker = diar_df.loc[best_idx, "speaker"] if best_idx is not None and best_ov >= min_overlap else "UNKNOWN"
        rows.append({
            "start": a.start,
            "end": a.end,
            "duration": a.duration,
            "speaker": speaker,
            "text": a.text,
            "confidence": a.confidence
        })
    return pd.DataFrame(rows)

def compute_turn_features(trans_df):
    # [Fix 15] Unchanged.
    if trans_df.empty:
        return pd.DataFrame()
    df = trans_df.sort_values("start").reset_index(drop=True).copy()
    df["next_start"] = df["start"].shift(-1)
    df["gap_to_next"] = df["next_start"] - df["end"]
    df["char_len"] = df["text"].fillna("").str.len()
    df["speech_rate_char_per_sec"] = df["char_len"] / df["duration"].replace(0, np.nan)
    return df

def compute_overlap_and_interruptions(turn_df, interruption_gap_threshold=0.3, min_overlap_sec=0.2):
    # [Fix 15] Unchanged.
    rows = []
    for i in range(len(turn_df)):
        for j in range(i + 1, len(turn_df)):
            a = turn_df.iloc[i]
            b = turn_df.iloc[j]
            ov = overlap(a.start, a.end, b.start, b.end)
            if ov >= min_overlap_sec and a.speaker != b.speaker:
                interrupter = None
                interrupted = None
                if b.start > a.start and b.start < a.end:
                    interrupter = b.speaker
                    interrupted = a.speaker
                rows.append({
                    "seg_a": i,
                    "seg_b": j,
                    "speaker_a": a.speaker,
                    "speaker_b": b.speaker,
                    "start": max(a.start, b.start),
                    "end": min(a.end, b.end),
                    "overlap_duration": ov,
                    "interrupter": interrupter,
                    "interrupted": interrupted
                })
    ov_df = pd.DataFrame(rows)
    if ov_df.empty:
        return ov_df, pd.DataFrame(columns=["speaker", "interruptions_made", "interruptions_received", "interruption_rate"])
    made = ov_df["interrupter"].value_counts().rename("interruptions_made")
    recv = ov_df["interrupted"].value_counts().rename("interruptions_received")
    spk = pd.Index(sorted(set(turn_df["speaker"]) - {"UNKNOWN"}), name="speaker")
    out = pd.DataFrame(index=spk).join(made).join(recv).fillna(0)
    turn_counts = turn_df.query("speaker != 'UNKNOWN'")["speaker"].value_counts().rename("turns")
    out = out.join(turn_counts).fillna(0)
    out["interruption_rate"] = out["interruptions_made"] / out["turns"].replace(0, np.nan)
    return ov_df, out.reset_index()

# ============================================================
# [Fix 15] Beep (masking tone) detection
# ============================================================

def detect_beep_sample_mask(
    y,
    sr=SR,
    frame_length=BEEP_FRAME_LENGTH,
    hop_length=BEEP_HOP_LENGTH,
    target_freq=BEEP_TARGET_FREQ,
    half_width=BEEP_HALF_WIDTH,
    band_ratio_thresh=BEEP_BAND_RATIO_THRESH,
    flatness_thresh=BEEP_FLATNESS_THRESH,
):
    """Return a per-sample beep mask (a bool array, True where the sample was
    judged to be a beep), computed once over the whole recording.

    The criteria come from measurements of a beep sample. Both have to hold at the
    same time, which keeps speech with harmonic structure, such as vowels, from
    being flagged:
      1. The energy ratio inside target_freq +/- half_width exceeds
         band_ratio_thresh (a beep concentrates its energy almost entirely on a
         single frequency).
      2. Spectral flatness is below flatness_thresh (a pure tone carries almost
         no noise component).
    """
    if len(y) < frame_length:
        return np.zeros(len(y), dtype=bool)

    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    band_mask = (freqs >= target_freq - half_width) & (freqs <= target_freq + half_width)

    band_energy = (S[band_mask] ** 2).sum(axis=0)
    total_energy = (S ** 2).sum(axis=0) + 1e-12
    band_ratio = band_energy / total_energy

    flatness = librosa.feature.spectral_flatness(
        y=y, n_fft=frame_length, hop_length=hop_length
    ).flatten()

    n_frames = min(len(band_ratio), len(flatness))
    frame_is_beep = (band_ratio[:n_frames] > band_ratio_thresh) & (flatness[:n_frames] < flatness_thresh)

    # Expand the per-frame decision to per-sample resolution.
    sample_mask = np.zeros(len(y), dtype=bool)
    for i, is_beep in enumerate(frame_is_beep):
        if not is_beep:
            continue
        s = i * hop_length
        e = min(s + frame_length, len(y))
        sample_mask[s:e] = True
    return sample_mask

def silence_out_beep(y, beep_mask, fade_samples=160):
    """[Fix 15-b] Return the audio with every sample judged to be a beep
    replaced by silence (0). The sample count and timeline are identical to the
    original, so start/end still refer to times in the original audio file.
    Feeding this signal to diarization and ASR keeps the beeps from affecting
    the transcript, the speech rate and the separation of speakers.
    A linear fade is applied over the fade_samples samples on either side of a
    beep stretch to avoid clicks at the boundaries.
    """
    y_clean = y.copy()
    y_clean[beep_mask] = 0.0

    if fade_samples <= 0:
        return y_clean

    # Find the beep stretches (runs of consecutive True) and fade each one in and out.
    diff = np.diff(beep_mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if beep_mask[0]:
        starts = np.insert(starts, 0, 0)
    if beep_mask[-1]:
        ends = np.append(ends, len(beep_mask))

    for s, e in zip(starts, ends):
        fade_out_start = max(0, s - fade_samples)
        if s > fade_out_start:
            ramp = np.linspace(1.0, 0.0, s - fade_out_start, endpoint=False)
            y_clean[fade_out_start:s] = y[fade_out_start:s] * ramp

        fade_in_end = min(len(y), e + fade_samples)
        if fade_in_end > e:
            ramp = np.linspace(0.0, 1.0, fade_in_end - e, endpoint=False)
            y_clean[e:fade_in_end] = y[e:fade_in_end] * ramp

    return y_clean

def extract_acoustic_features_for_segment(y, sr, start, end, beep_mask=None):
    s = int(start * sr)
    e = int(end * sr)
    seg = y[s:e]
    if len(seg) < sr * 0.2:
        return {}

    # [Fix 15] Drop the beep stretches before computing the acoustic features.
    if beep_mask is not None:
        seg_beep_mask = beep_mask[s:e]
        beep_ratio = float(seg_beep_mask.mean()) if len(seg_beep_mask) else 0.0
        seg_clean = seg[~seg_beep_mask]
    else:
        beep_ratio = 0.0
        seg_clean = seg

    if len(seg_clean) < sr * BEEP_MIN_CLEAN_SEC:
        # Almost no clean audio is left (the turn is nearly all beep), so the
        # acoustic features become NaN, but beep_ratio is always recorded for
        # diagnostics. The turn itself (duration/speaker/text) is kept by the
        # caller, so nothing here turns it into silence.
        return {
            "rms_mean": np.nan, "rms_std": np.nan, "zcr_mean": np.nan,
            "spectral_centroid_mean": np.nan,
            "f0_mean": np.nan, "f0_std": np.nan, "f0_min": np.nan,
            "f0_max": np.nan, "f0_range": np.nan,
            "beep_ratio": beep_ratio,
        }

    rms = librosa.feature.rms(y=seg_clean).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=seg_clean).flatten()
    centroid = librosa.feature.spectral_centroid(y=seg_clean, sr=sr).flatten()

    snd = parselmouth.Sound(seg_clean, sampling_frequency=sr)
    try:
        pitch = snd.to_pitch()
        f0 = pitch.selected_array["frequency"]
        f0 = f0[f0 > 0]
        f0_mean = float(np.mean(f0)) if len(f0) else np.nan
        f0_std = float(np.std(f0)) if len(f0) else np.nan
        f0_min = float(np.min(f0)) if len(f0) else np.nan
        f0_max = float(np.max(f0)) if len(f0) else np.nan
        f0_range = f0_max - f0_min if len(f0) else np.nan
    except Exception:
        f0_mean = f0_std = f0_min = f0_max = f0_range = np.nan

    return {
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zcr_mean": float(np.mean(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_range": f0_range,
        "beep_ratio": beep_ratio,
    }

def attach_acoustic_features(turn_df, y, sr, beep_mask=None):
    feats = []
    for _, r in turn_df.iterrows():
        d = extract_acoustic_features_for_segment(y, sr, r.start, r.end, beep_mask=beep_mask)
        feats.append(d)
    feat_df = pd.DataFrame(feats)
    return pd.concat([turn_df.reset_index(drop=True), feat_df], axis=1)

def aggregate_speaker_stats(turn_df, silence_df, total_dur):
    base = turn_df.query("speaker != 'UNKNOWN'").copy()
    if base.empty:
        return pd.DataFrame()
    grp = base.groupby("speaker")
    out = pd.DataFrame({
        "turn_count": grp.size(),
        "speech_total_sec": grp["duration"].sum(),
        "speech_mean_sec": grp["duration"].mean(),
        "speech_median_sec": grp["duration"].median(),
        "char_total": grp["char_len"].sum(),
        "speech_rate_char_per_sec_mean": grp["speech_rate_char_per_sec"].mean(),
        "rms_mean": grp["rms_mean"].mean(),
        "f0_mean": grp["f0_mean"].mean(),
        "f0_std_mean": grp["f0_std"].mean(),
        "f0_range_mean": grp["f0_range"].mean(),
        "beep_ratio_mean": grp["beep_ratio"].mean() if "beep_ratio" in base.columns else np.nan,
    }).reset_index()
    out["share_of_meeting_time"] = out["speech_total_sec"] / total_dur
    return out.sort_values("speech_total_sec", ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="input wav path")
    ap.add_argument("--outdir", required=True, help="output directory")
    ap.add_argument("--hf_token", default=os.getenv("HF_TOKEN"))
    ap.add_argument("--language", default="ja")
    # [Fix 16] Default to large-v3, matching the batch driver used in practice
    # (02_run_pipeline_batch.py). Leaving it at small produced far more hallucinations
    # (looping output) when the script was run on its own.
    ap.add_argument("--whisper_model", default="large-v3")
    args = ap.parse_args()

    # [Fix 16] Without HF_TOKEN, diarization is skipped silently and every turn
    # ends up as speaker=UNKNOWN. Warn explicitly so that nobody uses such an
    # output without noticing.
    if not args.hf_token:
        print(
            "[INFO] No HF_TOKEN was found in the environment or in --hf_token. "
            "If huggingface_hub has cached login credentials (huggingface-cli login), "
            "we fall back to those and still attempt speaker diarization. "
            "If that fallback also fails, '[WARN] diarization pipeline failed' is "
            "printed and every turn becomes speaker=UNKNOWN."
        )

    ensure_dir(args.outdir)
    y = load_audio_mono(args.audio, SR)
    total_dur = len(y) / SR

    # [Fix 15] Compute the beep mask once over the whole recording.
    beep_mask = detect_beep_sample_mask(y, SR)
    beep_total_sec = float(beep_mask.sum() / SR)
    print(f"[INFO] Total time judged to be beeps: {beep_total_sec:.2f} s "
          f"({beep_total_sec / total_dur * 100:.2f}% of recording)")

    # [Fix 15-b] Build the audio with the beep stretches silenced; its timeline
    # matches the original. Diarization and ASR take this as their input so that
    # speech rate, transcript and speaker separation stay free of beep
    # contamination.
    y_for_diar_asr = silence_out_beep(y, beep_mask)

    speech_df = detect_speech_regions_vad(y_for_diar_asr, SR)
    silence_df = complement_silence_regions(total_dur, speech_df)

    # [Fix 16] Diarization used to be skipped entirely when args.hf_token was
    # empty, which meant that simply forgetting to export HF_TOKEN silently
    # turned every turn into speaker=UNKNOWN. huggingface_hub falls back to the
    # token cached by `huggingface-cli login` even when use_auth_token=None
    # (verified to work with the cached token on this machine), so diarize_audio
    # is now always called.
    diar_df = diarize_audio(args.audio, args.hf_token, waveform=y_for_diar_asr, sr=SR)
    asr_df = transcribe_audio(args.audio, args.whisper_model, args.language, waveform=y_for_diar_asr, total_dur=total_dur)

    if not diar_df.empty:
        turn_df = assign_text_to_speakers(diar_df, asr_df)
    else:
        turn_df = asr_df.copy()
        turn_df["speaker"] = "UNKNOWN"

    turn_df = compute_turn_features(turn_df)
    turn_df = attach_acoustic_features(turn_df, y, SR, beep_mask=beep_mask)

    overlap_df, interruption_df = compute_overlap_and_interruptions(turn_df)
    speaker_stats_df = aggregate_speaker_stats(turn_df, silence_df, total_dur)

    n_turns_beep_affected = int((turn_df["beep_ratio"] > 0.05).sum()) if "beep_ratio" in turn_df.columns else 0
    n_turns_beep_dominant = int((turn_df["beep_ratio"] > 0.5).sum()) if "beep_ratio" in turn_df.columns else 0

    meeting_summary = {
        "audio_path": args.audio,
        "duration_sec": total_dur,
        "speech_total_sec": float(speech_df["duration"].sum()) if not speech_df.empty else 0.0,
        "silence_total_sec": float(silence_df["duration"].sum()) if not silence_df.empty else 0.0,
        "silence_ratio": float(silence_df["duration"].sum() / total_dur) if total_dur > 0 else np.nan,
        "speaker_count_estimated": int(turn_df.query("speaker != 'UNKNOWN'")["speaker"].nunique()) if "speaker" in turn_df else 0,
        "turn_count": int(len(turn_df)),
        "overlap_count": int(len(overlap_df)),
        "overlap_total_sec": float(overlap_df["overlap_duration"].sum()) if not overlap_df.empty else 0.0,
        # [Fix 15] Beep-related diagnostics.
        "beep_total_sec": beep_total_sec,
        "beep_ratio_of_recording": float(beep_total_sec / total_dur) if total_dur > 0 else np.nan,
        "turns_beep_affected_gt5pct": n_turns_beep_affected,
        "turns_beep_dominant_gt50pct": n_turns_beep_dominant,
        # [Fix 16] Record the run parameters so that it stays possible to tell
        # afterwards which settings produced an output, and to catch runs made
        # without a token or with too small a model. The token value itself is
        # never stored.
        "whisper_model_used": args.whisper_model,
        "hf_token_provided": bool(args.hf_token),
    }

    turn_df.to_csv(Path(args.outdir) / "turns_with_text_and_acoustics.csv", index=False)
    speech_df.to_csv(Path(args.outdir) / "speech_regions_vad.csv", index=False)
    silence_df.to_csv(Path(args.outdir) / "silence_regions.csv", index=False)
    diar_df.to_csv(Path(args.outdir) / "speaker_diarization_segments.csv", index=False)
    asr_df.to_csv(Path(args.outdir) / "asr_segments.csv", index=False)
    overlap_df.to_csv(Path(args.outdir) / "overlap_and_interruptions.csv", index=False)
    interruption_df.to_csv(Path(args.outdir) / "speaker_interruption_stats.csv", index=False)
    speaker_stats_df.to_csv(Path(args.outdir) / "speaker_summary_stats.csv", index=False)

    with open(Path(args.outdir) / "meeting_summary.json", "w", encoding="utf-8") as f:
        json.dump(meeting_summary, f, ensure_ascii=False, indent=2)

    transcript_txt = []
    for _, r in turn_df.iterrows():
        transcript_txt.append(f"[{r.start:8.2f} - {r.end:8.2f}] {r.speaker}: {r.text}")
    Path(args.outdir, "speaker_attributed_transcript.txt").write_text("\n".join(transcript_txt), encoding="utf-8")

if __name__ == "__main__":
    main()
