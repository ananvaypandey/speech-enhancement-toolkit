"""Ad-hoc calibration check for the K-weighting / LUFS implementation.

Not part of the pytest suite: this prints a table so the numbers can be
inspected by eye against the ITU-R BS.1770 calibration point.
Run: .venv\\Scripts\\python.exe scripts\\check_lufs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from aelf.analysis.levels import integrated_lufs
from aelf.types import AudioBuffer

SR = 48_000


def tone(freq: float, seconds: float, sr: int, amplitude: float) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def main() -> int:
    print("ITU-R BS.1770 calibration: a 1 kHz sine reads peak_dBFS - 3.01 LUFS\n")
    print(f"{'peak':>8} {'measured':>12} {'expected':>12} {'error dB':>10}")
    worst = 0.0
    for amp_db in (0.0, -6.0, -20.0, -30.0, -40.0):
        amp = 10.0 ** (amp_db / 20.0)
        buf = AudioBuffer(samples=tone(1000.0, 10.0, SR, amp), sample_rate=SR)
        lufs, _lra = integrated_lufs(buf)
        expected = amp_db - 3.01
        error = lufs - expected
        worst = max(worst, abs(error))
        print(f"{amp_db:>7.1f}dB {lufs:>11.2f}  {expected:>11.2f}  {error:>9.3f}")

    print("\nRate independence (1 kHz full-scale sine, expect -3.01 LUFS):")
    for sr in (16_000, 22_050, 44_100, 48_000, 96_000):
        buf = AudioBuffer(samples=tone(1000.0, 10.0, sr, 1.0), sample_rate=sr)
        lufs, _ = integrated_lufs(buf)
        err = lufs - (-3.01)
        worst = max(worst, abs(err))
        print(f"  {sr:>6} Hz -> {lufs:8.2f} LUFS  (err {err:+.3f})")

    print("\nSpectral tilt (1 kHz sine at +3 dB of HF weighting):")
    for freq in (100.0, 1000.0, 4000.0, 10_000.0):
        buf = AudioBuffer(samples=tone(freq, 10.0, SR, 1.0), sample_rate=SR)
        lufs, _ = integrated_lufs(buf)
        print(f"  {freq:>8.0f} Hz -> {lufs:8.2f} LUFS")

    print(f"\nWorst calibration error: {worst:.3f} dB")
    return 0 if worst < 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
