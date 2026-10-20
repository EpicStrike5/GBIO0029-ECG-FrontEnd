#!/usr/bin/env python3
"""
Synthetic ECG and EOG stimulus generator for LTspice (PWL files).

Pure Python standard library, no numpy needed. Run:
    python make_stimuli.py
and the .txt files are (re)written in the same folder as this script.

Every file is plain text, one "time value" pair per line, seconds and volts,
which LTspice reads with a voltage source set to  PWL file=<name>.txt

Files produced for each signal, for ONE voltage source between the
amplifier inputs:
    ecg_clean.txt         ideal ECG, lead II like, ~1 mV R peak
    ecg_noisy_diff.txt    + baseline wander + differential 50 Hz leakage
                          + drift + small broadband noise (offset: DC_OFFSET_MV)
    eog_clean.txt, eog_noisy_diff.txt   same idea for horizontal EOG
    ecg_wander_noise.txt  ECG + breathing wander + white noise only
    eog_wander_noise.txt  (no offset, no 50 Hz: use these with the body
                          model in the schematic, which adds those itself)

All parameters are in the PARAMETERS block; change them and run again.
"""

import math
import os
import random

# ----------------------------------------------------------------- PARAMETERS
FS = 2000.0            # sample rate in Hz (2 kHz = 40 points per 50 Hz cycle)
DURATION = 30.0        # seconds
SEED = 1               # change for a different random realisation

MAINS_HZ = 50.0        # 50 in Belgium, 60 in the Americas

# --- ECG
ECG_BPM = 70.0
ECG_HRV = 0.03         # beat-to-beat RR jitter, fraction of RR (3 %)
ECG_R_MV = 1.0         # R peak amplitude in mV
# Gaussian waves relative to the R peak: (name, centre s, amplitude mV, width s)
# Amplitudes are scaled so the R wave equals ECG_R_MV.
ECG_WAVES = [
    ("P", -0.200, 0.15, 0.025),
    ("Q", -0.028, -0.12, 0.009),
    ("R",  0.000, 1.00, 0.011),
    ("S",  0.030, -0.25, 0.010),
    ("T",  0.290, 0.30, 0.055),
]

# --- EOG (horizontal, electrodes at both outer canthi)
EOG_UV_PER_DEG = 16.0  # typical 10..20 uV per degree of gaze angle
EOG_MAX_DEG = 25.0     # saccade targets drawn within +/- this angle
EOG_FIX_S = (0.3, 1.2) # fixation duration range, seconds
EOG_BLINKS = 3         # blink artefacts (they leak into horizontal leads)
EOG_BLINK_UV = 60.0    # blink amplitude seen on the horizontal lead

# --- Interference (applied to the *_noisy_diff files)
WANDER_UV = 150.0      # respiration baseline wander, peak, differential
WANDER_HZ = 0.25       # ~15 breaths per minute
DRIFT_UV_PER_S = 20.0  # slow electrode drift, differential
DC_OFFSET_MV = 0.0     # electrode offset, 0 = add it in LTspice instead (worst case 100)
MAINS_DIFF_UV = 30.0   # differential 50 Hz that leaks in, peak
MAINS_HARM3 = 0.2      # 3rd harmonic relative to the fundamental
NOISE_UV_RMS = 5.0     # broadband noise (EMG + electrode noise), rms

# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(round(DURATION * FS)) + 1
T = [i / FS for i in range(N)]


def write_pwl(name, values):
    path = os.path.join(HERE, name)
    with open(path, "w", newline="\n") as f:
        for t, v in zip(T, values):
            f.write(f"{t:.6f} {v:.6e}\n")
    print(f"wrote {name}  ({len(values)} points)")


def ecg_clean(rng):
    """Sum of Gaussians per beat, with heart-rate variability."""
    rr0 = 60.0 / ECG_BPM
    scale = ECG_R_MV / max(a for _, _, a, _ in ECG_WAVES)
    beats = []
    t = 0.35  # first R peak
    while t < DURATION + 1.0:
        beats.append(t)
        t += rr0 * (1.0 + rng.gauss(0.0, ECG_HRV))
    y = [0.0] * N
    for r in beats:
        for _, c, a, w in ECG_WAVES:
            centre = r + c
            lo = max(0, int((centre - 5 * w) * FS))
            hi = min(N, int((centre + 5 * w) * FS) + 1)
            for i in range(lo, hi):
                d = (T[i] - centre) / w
                y[i] += a * scale * math.exp(-0.5 * d * d)
    return [v * 1e-3 for v in y]  # mV -> V


def eog_clean(rng):
    """Fixations joined by saccades with a realistic duration (main sequence)."""
    y = [0.0] * N
    angle = 0.0
    t = 0.5
    events = []  # (start time, from angle, to angle, duration)
    while t < DURATION:
        target = rng.uniform(-EOG_MAX_DEG, EOG_MAX_DEG)
        amp = abs(target - angle)
        dur = 0.0022 * amp + 0.021  # saccade main sequence, ~2.2 ms/deg + 21 ms
        events.append((t, angle, target, dur))
        angle = target
        t += dur + rng.uniform(*EOG_FIX_S)
    angle = 0.0
    k = 0
    for i in range(N):
        while k < len(events) and T[i] >= events[k][0] + events[k][3]:
            angle = events[k][2]
            k += 1
        if k < len(events) and T[i] >= events[k][0]:
            t0, a0, a1, dur = events[k]
            x = (T[i] - t0) / dur
            s = 0.5 - 0.5 * math.cos(math.pi * x)  # smooth S-shaped saccade
            y[i] = a0 + (a1 - a0) * s
        else:
            y[i] = angle
    y = [a * EOG_UV_PER_DEG for a in y]
    for _ in range(EOG_BLINKS):
        tb = rng.uniform(1.0, DURATION - 0.5)
        lo, hi = int((tb - 0.3) * FS), int((tb + 0.3) * FS)
        for i in range(max(0, lo), min(N, hi)):
            d = (T[i] - tb) / 0.06
            y[i] += EOG_BLINK_UV * math.exp(-0.5 * d * d)
    return [v * 1e-6 for v in y]  # uV -> V


def diff_interference(rng):
    w = 2 * math.pi
    ph = rng.uniform(0, w)
    out = []
    for t in T:
        v = DC_OFFSET_MV * 1e-3
        v += WANDER_UV * 1e-6 * math.sin(w * WANDER_HZ * t + ph)
        v += DRIFT_UV_PER_S * 1e-6 * t
        v += MAINS_DIFF_UV * 1e-6 * (math.sin(w * MAINS_HZ * t)
                                     + MAINS_HARM3 * math.sin(3 * w * MAINS_HZ * t))
        v += rng.gauss(0.0, NOISE_UV_RMS * 1e-6)
        out.append(v)
    return out


def wander_noise(rng):
    """Breathing wander + white noise only (no offset, no drift, no 50 Hz)."""
    w = 2 * math.pi
    ph = rng.uniform(0, w)
    return [WANDER_UV * 1e-6 * math.sin(w * WANDER_HZ * t + ph)
            + rng.gauss(0.0, NOISE_UV_RMS * 1e-6) for t in T]


def emit(prefix, clean, rng):
    noisy = [c + n for c, n in zip(clean, diff_interference(rng))]
    write_pwl(f"{prefix}_clean.txt", clean)
    write_pwl(f"{prefix}_noisy_diff.txt", noisy)
    # separate random stream so the files above stay identical
    rng2 = random.Random(f"{SEED}-{prefix}")
    write_pwl(f"{prefix}_wander_noise.txt",
              [c + n for c, n in zip(clean, wander_noise(rng2))])


if __name__ == "__main__":
    rng = random.Random(SEED)
    emit("ecg", ecg_clean(rng), rng)
    emit("eog", eog_clean(rng), rng)
