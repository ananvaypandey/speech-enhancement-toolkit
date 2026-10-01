"""Compare our LUFS implementation against pyloudnorm.

pyloudnorm is an independent implementation of ITU-R BS.1770-4, so agreement
with it is meaningful evidence that our K-weighting and gating are right.
Run: .venv\\Scripts\\python.exe scripts\\check_lufs_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from aelf.analysis.levels import integrated_lufs
from aelf.types import AudioBuffer

SR = 48_000
TOLERANCE_DB = 0.2


def tone(freq: float, seconds: float, sr: int, amplitude: float) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def speech_like(seconds: float, sr: int, f0: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * seconds)
    t = np.arange(n) / sr
    sig = np.zeros(n)
    for k in range(1, 26):
        f = f0 * k
        if f > sr / 2.2:
            break
        g = (1.0 / k) * (1.0 + 1.6 * np.exp(-((f - 700.0) ** 2) / (2 * 300.0**2)))
        sig += g * np.sin(2.0 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    sig /= np.max(np.abs(sig)) or 1.0
    env = np.clip(0.5 + 0.5 * np.sin(2.0 * np.pi * 4.0 * t), 0.0, 1.0) ** 1.5
    return (sig * env + rng.normal(0.0, 0.01, n)).astype(np.float32) * 0.3


def main() -> int:
    print("pyloudnorm 'DeMan' filters are the spec-accurate BS.1770 coefficients.")
    print("Its default 'K-weighting' is an RBJ approximation (1500 Hz shelf) and")
    print("is known to sit ~0.6 dB off, so it is shown but not used as truth.\n")

    meter_de_man = pyln.Meter(SR, filter_class="DeMan")
    meter_approx = pyln.Meter(SR, filter_class="K-weighting")
    cases: list[tuple[str, np.ndarray]] = [
        ("sine 1k full scale", tone(1000.0, 10.0, SR, 1.0)),
        ("sine 1k -20 dBFS", tone(1000.0, 10.0, SR, 0.1)),
        ("sine 1k -40 dBFS", tone(1000.0, 10.0, SR, 0.01)),
        ("sine 100 Hz", tone(100.0, 10.0, SR, 0.5)),
        ("sine 4 kHz", tone(4000.0, 10.0, SR, 0.5)),
        ("speech-like", speech_like(10.0, SR, 120.0, 3)),
        ("speech-like quiet", speech_like(10.0, SR, 120.0, 3) * 0.05),
    ]

    print(f"{'case':<22} {'ours':>9} {'DeMan':>9} {'delta':>8} {'RBJ approx':>11}")
    print("-" * 64)
    worst = 0.0
    for name, data in cases:
        ours, _ = integrated_lufs(AudioBuffer(samples=data, sample_rate=SR))
        truth = meter_de_man.integrated_loudness(data.astype(np.float64))
        approx = meter_approx.integrated_loudness(data.astype(np.float64))
        delta = ours - truth
        worst = max(worst, abs(delta))
        print(f"{name:<22} {ours:>8.2f}  {truth:>8.2f}  {delta:>+7.3f}  {approx:>10.2f}")

    print("-" * 64)
    print(f"worst deviation vs spec-accurate reference: {worst:.3f} dB (tolerance {TOLERANCE_DB} dB)")
    return 0 if worst < TOLERANCE_DB else 1


if __name__ == "__main__":
    raise SystemExit(main())
