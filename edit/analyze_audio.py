#!/usr/bin/env python3
"""Analyse the music track and write a beat map the timeline is built on.

Output JSON:
  tempo            global BPM estimate
  beats            beat times (s)
  downbeats        every 4th beat, phase chosen by bass energy
  bass_hits        strong low-frequency (808/kick) transients, with strength 0..1
  accents          strong broadband onsets (snares, cowbells, stabs), strength 0..1
  energy           RMS envelope sampled at 10 Hz (0..1)
  sections         [start, end, mean_energy] split at large energy changes
  silences         gaps where the track drops out (pauses / pre-drop breaks)

Usage: analyze_audio.py music.wav beats.json [--start S] [--duration D]
"""
import argparse
import json

import librosa
import numpy as np


def norm(x):
    x = np.asarray(x, dtype=np.float64)
    r = x.max() - x.min()
    return (x - x.min()) / r if r > 0 else np.zeros_like(x)


def peaks(env, times, delta, wait):
    idx = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=12, post_avg=12,
                                 delta=delta, wait=wait)
    return [(float(times[i]), float(env[i])) for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("out")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    a = ap.parse_args()

    y, sr = librosa.load(a.audio, sr=44100, mono=True, offset=a.start, duration=a.duration)
    hop = 256
    dur = len(y) / sr

    # Beat grid. Slowed phonk often reads at half/double tempo; keep it in 60..160.
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    # Band energies.
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    low = S[f < 150].sum(0)
    low_flux = norm(np.maximum(0, np.diff(low, prepend=low[0])))
    high_flux = norm(librosa.onset.onset_strength(S=librosa.amplitude_to_db(S[f > 1500]),
                                                  sr=sr, hop_length=hop))
    min_gap = int(0.12 * sr / hop)
    bass_hits = peaks(low_flux, t, 0.12, min_gap)
    accents = peaks(high_flux, t, 0.15, min_gap)

    # The tracker sometimes locks onto hi-hats between kicks. Re-phase the grid onto the
    # bass, extend it over the whole track, then snap each beat to a nearby bass hit.
    period = float(np.median(np.diff(beats))) if len(beats) > 2 else 60.0 / tempo
    shifts = np.linspace(-period / 2, period / 2, 33)
    score = [np.interp(beats + s, t, low_flux).sum() for s in shifts]
    beats = beats + shifts[int(np.argmax(score))]
    first = max(0.0, beats[0] - period * np.floor((beats[0] + 0.05) / period))
    grid = list(np.arange(first, beats[0] - period / 2, period)) + list(beats)
    while grid[-1] + period < dur:
        grid.append(grid[-1] + period)
    hit_t = np.array([h for h, s in bass_hits if s > 0.25])
    beats = []
    for b in grid:
        if len(hit_t):
            j = np.argmin(np.abs(hit_t - b))
            if abs(hit_t[j] - b) < 0.06:
                b = hit_t[j]
        beats.append(b)
    beats = np.array(beats)
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop)

    # Downbeat phase: the beat offset (0..3) whose beats carry the most bass.
    low_n = norm(low)
    bi = np.clip(beat_frames, 0, len(low_n) - 1)
    phase = int(np.argmax([low_n[bi[p::4]].mean() if len(bi[p::4]) else 0 for p in range(4)]))
    downbeats = beats[phase::4]

    # Energy envelope at 10 Hz, sections and silences.
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    grid = np.arange(0, dur, 0.1)
    energy = norm(np.interp(grid, rms_t, rms))
    smooth = np.convolve(energy, np.ones(10) / 10, mode="same")
    cuts = [0.0]
    for i in range(10, len(smooth) - 10):
        if abs(smooth[i + 5] - smooth[i - 5]) > 0.25 and grid[i] - cuts[-1] > 3:
            cuts.append(float(grid[i]))
    cuts.append(dur)
    sections = [[s, e, float(energy[(grid >= s) & (grid < e)].mean())]
                for s, e in zip(cuts[:-1], cuts[1:])]
    quiet = energy < 0.08
    silences, start = [], None
    for g, q in zip(grid, quiet):
        if q and start is None:
            start = g
        elif not q and start is not None:
            if g - start >= 0.2:
                silences.append([float(start), float(g)])
            start = None

    out = {
        "file": a.audio, "offset": a.start, "duration": dur, "tempo": tempo,
        "beats": [round(float(b), 4) for b in beats],
        "downbeats": [round(float(b), 4) for b in downbeats],
        "bass_hits": [[round(x, 4), round(s, 3)] for x, s in bass_hits],
        "accents": [[round(x, 4), round(s, 3)] for x, s in accents],
        "energy": [round(float(e), 3) for e in energy],
        "sections": sections, "silences": silences,
    }
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"tempo {tempo:.1f} BPM, {len(beats)} beats, {len(bass_hits)} bass hits, "
          f"{len(accents)} accents, {len(sections)} sections, {len(silences)} silences, "
          f"{dur:.2f}s")


if __name__ == "__main__":
    main()
