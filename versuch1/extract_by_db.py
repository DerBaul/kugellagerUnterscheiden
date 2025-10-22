#!/usr/bin/env python3
"""
extract_by_db.py

Teilt eine Audiodatei an allen Stellen, an denen der Pegel unter einen
absoluten dBFS-Grenzwert fällt (z.B. -45 dB). Kurze stille Lücken können
automatisch "geschlossen" (nicht als Schnitt) werden, sodass nur längere
Absenkungen als Trennungen genutzt werden.

Abhängigkeiten:
- numpy
- librosa
- soundfile

Installation:
pip install numpy librosa soundfile

Beispiel:
python extract_by_db.py input.wav --threshold_db -45 --merge_gap_ms 30 --keep_silence_ms 8 --outdir clips

Hinweis:
Dieses Update überspringt zusätzlich alle Clips, die kürzer als 1 Sekunde sind
(werden nicht gespeichert).
"""
import argparse
from pathlib import Path
import os

import numpy as np
import librosa
import soundfile as sf


def load_mono(path, sr=None):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def rms_db_by_frames(y, sr, frame_length, hop_length):
    # RMS pro frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    # In dBFS (librosa amplitudes sind in [-1,1], ref=1.0 -> dBFS)
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    return rms_db


def close_small_gaps(mask, max_gap_frames):
    """
    mask: boolean array True=signal über Schwellwert
    ersetzt kurze False-Runs (<= max_gap_frames) durch True
    """
    if max_gap_frames <= 0:
        return mask
    mask_int = mask.astype(np.int8)
    n = len(mask_int)
    i = 0
    while i < n:
        if mask_int[i] == 0:
            # start of false run
            j = i
            while j < n and mask_int[j] == 0:
                j += 1
            run_len = j - i
            if run_len <= max_gap_frames:
                # close the gap
                mask_int[i:j] = 1
            i = j
        else:
            i += 1
    return mask_int.astype(bool)


def frames_to_intervals(mask):
    """
    mask: boolean per-frame -> return list of (start_frame, end_frame) with end exclusive
    """
    if len(mask) == 0:
        return []
    diffs = np.diff(mask.astype(np.int8))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1

    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [len(mask)]))

    intervals = list(zip(starts.tolist(), ends.tolist()))
    return intervals


def intervals_frames_to_samples(intervals, hop_length):
    # frames -> sample indices (frame start sample)
    samp_intervals = []
    for s_f, e_f in intervals:
        s_s = librosa.frames_to_samples(s_f, hop_length=hop_length)
        e_s = librosa.frames_to_samples(e_f, hop_length=hop_length)
        samp_intervals.append((int(s_s), int(e_s)))
    return samp_intervals


def extract_by_db(path, outdir="clips", threshold_db=-45.0, frame_ms=10.0, hop_ms=None,
                  merge_gap_ms=30.0, keep_silence_ms=8.0, min_duration_ms=0.0):
    y, sr = load_mono(path, sr=None)
    print(f"Lade '{path}': {len(y)} Samples, SR={sr}")

    frame_length = max(1, int(sr * frame_ms / 1000))
    if hop_ms is None:
        hop_length = max(1, frame_length // 2)
        hop_ms = hop_length / sr * 1000.0
    else:
        hop_length = max(1, int(sr * hop_ms / 1000))

    print(f"Frame {frame_length} samples ({frame_ms} ms), Hop {hop_length} samples (~{hop_ms:.1f} ms)")

    rms_db = rms_db_by_frames(y, sr, frame_length=frame_length, hop_length=hop_length)

    # Maske: True = oberhalb Schwellwert (signal), False = "stille" (unter threshold_db)
    mask = rms_db > threshold_db

    # Schließe sehr kurze stille Lücken (in frames)
    max_gap_frames = int(round(merge_gap_ms / (hop_length / sr * 1000.0))) if merge_gap_ms is not None else 0
    if max_gap_frames < 0:
        max_gap_frames = 0
    mask_closed = close_small_gaps(mask, max_gap_frames=max_gap_frames)

    # Bestimme Intervalle in frames und konvertiere zu Samples
    frame_intervals = frames_to_intervals(mask_closed)
    sample_intervals = intervals_frames_to_samples(frame_intervals, hop_length=hop_length)

    # Anwenden von keep_silence und min_duration
    keep = int(round(sr * keep_silence_ms / 1000.0))
    min_samples = int(round(sr * min_duration_ms / 1000.0))

    out_paths = []
    base = Path(path).stem
    os.makedirs(outdir, exist_ok=True)

    idx = 0
    skipped_too_short_under_min = 0
    skipped_too_short_under_1s = 0

    for (s, e) in sample_intervals:
        s2 = max(0, s - keep)
        e2 = min(len(y), e + keep)
        length = e2 - s2

        # neues Verhalten: lösche / speichere nicht Clips < 1 Sekunde
        if length < sr:
            skipped_too_short_under_1s += 1
            continue

        # bereits vorhandene min_duration_ms Regel (weiterhin gültig)
        if length < min_samples:
            skipped_too_short_under_min += 1
            continue

        seg = y[s2:e2]
        outname = Path(outdir) / f"{base}_{idx:03d}.wav"
        sf.write(str(outname), seg, sr, subtype="PCM_16")
        out_paths.append(str(outname))
        idx += 1

    print(f"Gefundene Intervalle: {len(sample_intervals)}")
    print(f"Übersprungen (Länge < 1s): {skipped_too_short_under_1s}")
    print(f"Übersprungen (Länge < min_duration_ms): {skipped_too_short_under_min}")
    print(f"Gespeichert: {len(out_paths)} in '{outdir}'")
    return out_paths


def main():
    parser = argparse.ArgumentParser(description="Split audio where level < threshold_db (absolute dBFS).")
    parser.add_argument("input", help="Eingabe-Audiodatei")
    parser.add_argument("--outdir", "-o", default="clips", help="Ausgabeverzeichnis")
    parser.add_argument("--threshold_db", type=float, default=-45.0, help="Schwelle in dBFS (z.B. -45)")
    parser.add_argument("--frame_ms", type=float, default=10.0, help="Frame-Größe in ms für RMS")
    parser.add_argument("--hop_ms", type=float, default=None, help="Hop-Größe in ms (default frame_ms/2)")
    parser.add_argument("--merge_gap_ms", type=float, default=30.0, help="Schließe stille Lücken <= X ms (nicht schneiden)")
    parser.add_argument("--keep_silence_ms", type=float, default=8.0, help="Kontext (ms) vor/nach jedem Segment behalten")
    parser.add_argument("--min_duration_ms", type=float, default=0.0, help="Minimale Segmentdauer (ms), kürzere verwerfen")
    args = parser.parse_args()

    extract_by_db(args.input, outdir=args.outdir, threshold_db=args.threshold_db,
                  frame_ms=args.frame_ms, hop_ms=args.hop_ms,
                  merge_gap_ms=args.merge_gap_ms, keep_silence_ms=args.keep_silence_ms,
                  min_duration_ms=args.min_duration_ms)


if __name__ == "__main__":
    main()