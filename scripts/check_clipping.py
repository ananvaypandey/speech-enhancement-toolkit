"""Verify that no stage in the pipeline can produce clipping.

Run after any change to the filter chain, EQ, limiter, or normaliser. A
regression here silently damages every processed file the toolkit writes.

Note on methodology: `sosfiltfilt` pads with an odd extension, so a filter run
over a buffer overshoots near the first and last few samples. That is an
artefact of the zero-phase implementation, not a defect in the signal path, so
stage-gain checks here measure the transfer function from the impulse response
instead of relying on time-domain peaks.
"""

from __future__ import annotations

import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, "src")
sys.path.insert(0, "tests")

from aelf.config import CLIP_CEILING_DBFS, CLIP_THRESHOLD, TRUE_PEAK_CEILING
from aelf.io.encode import count_clipped, true_peak_limit
from aelf.postprocess.filters import apply_filter_chain, highpass, peaking_eq, speech_band_eq
from aelf.postprocess.normalize import normalise_lufs
from aelf.types import AudioBuffer
from conftest import SR, sibilant_burst, speech_like, tone, white_noise

IR_N = 16_384


def db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-12))


def report(label: str, x: np.ndarray, ceiling: float) -> bool:
    peak = float(np.max(np.abs(x)))
    clipped = count_clipped(x)
    ok = clipped == 0 and peak <= ceiling + 1e-6
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<44} peak={peak:.4f} ({db(peak):+.2f} dBFS)  clipped={clipped}")
    return ok


def tf_gain_db(out: np.ndarray, at_hz: float, sr: int = SR) -> float:
    """Magnitude response of a filter at one frequency, from its impulse response."""
    spectrum = np.abs(np.fft.rfft(out.astype(np.float64), IR_N * 4))
    freqs = np.fft.rfftfreq(IR_N * 4, 1.0 / sr)
    idx = int(np.argmin(np.abs(freqs - at_hz)))
    return db(spectrum[idx])


def check(label: str, fn, at_hz: float, expect_db: float, tol: float = 0.05) -> bool:
    imp = np.zeros(IR_N, dtype=np.float32)
    imp[IR_N // 2] = 1.0  # mid-buffer: odd-extension padding cannot mirror it
    got = tf_gain_db(fn(imp), at_hz)
    ok = abs(got - expect_db) <= tol
    print(f"  {'OK  ' if ok else 'FAIL'} {label:<44} {got:+7.2f} dB (expected {expect_db:+.2f} +/- {tol})")
    return ok


def main() -> int:
    print(f"ceiling: {CLIP_CEILING_DBFS} dBFS = {TRUE_PEAK_CEILING:.6f} linear")
    print(f"clip threshold: {CLIP_THRESHOLD}")
    print()
    failures = 0

    print("filter gain accuracy (impulse-response transfer function):")
    # `peaking_eq` is measured directly rather than through `speech_band_eq`,
    # because the latter ends with the limiter, which scales the response.
    failures += not check("highpass @ 1000 Hz (expect 0 dB)", lambda x: highpass(x, SR, 80.0)[0], 1000.0, 0.0)
    # Zero-phase filtering applies the design twice, so the effective order is
    # doubled: 2nd-order Butterworth at the cutoff becomes -6 dB, not -3 dB.
    failures += not check("highpass @ 80 Hz (expect -6 dB)", lambda x: highpass(x, SR, 80.0)[0], 80.0, -6.0, 0.2)
    for g in (6.0, 3.0, -3.0):
        failures += not check(
            f"peaking EQ {g:+.0f} dB @ 2.8 kHz", lambda x, g=g: peaking_eq(x, SR, center_hz=2800.0, gain_db=g, q=0.8),
            2800.0, g,
        )

    print()
    print("EQ stage holds the ceiling on loud material:")
    loud_sources = {
        "full-scale tone": tone(2800.0, 1.0, SR, 0.99),
        "full-scale speech-like": (speech_like(2.0, SR, 120.0, 4) * 3.0).astype(np.float32),
        "white noise sigma 0.4": white_noise(2.0, SR, 0.4, seed=9),
    }
    for name, x in loud_sources.items():
        out, _ = speech_band_eq(x, SR, presence_boost_db=6.0, mud_cut_db=-3.0)
        failures += not report(name, out, TRUE_PEAK_CEILING)

    print()
    print("full filter chain (EQ + de-esser + limiter):")
    chain_sources = dict(loud_sources)
    chain_sources["sibilant burst"] = sibilant_burst()
    for name, x in chain_sources.items():
        buf = AudioBuffer(samples=x[:, None], sample_rate=SR)
        result = apply_filter_chain(buf)
        failures += not report(name, result.audio.samples[:, 0], TRUE_PEAK_CEILING)

    print()
    print("LUFS normalisation to +0 dBFS target (limiter must catch it):")
    for name, x in chain_sources.items():
        buf = AudioBuffer(samples=x[:, None], sample_rate=SR)
        out = normalise_lufs(buf, target_lufs=0.0)
        failures += not report(name, out.samples[:, 0], TRUE_PEAK_CEILING)

    print()
    print("limiter is exact:")
    loud = np.array([2.0, -2.0, 1.0, -1.0], dtype=np.float32)
    limited = true_peak_limit(loud, CLIP_CEILING_DBFS)
    peak = float(np.max(np.abs(limited)))
    ok = peak <= TRUE_PEAK_CEILING + 1e-6 and abs(peak - TRUE_PEAK_CEILING) < 1e-5
    print(f"  {'OK  ' if ok else 'FAIL'} clamps +/-2.0 -> {peak:.6f} (target {TRUE_PEAK_CEILING:.6f})")
    failures += not ok

    print()
    print("band-passed HF burst stays bounded:")
    sos = butter(2, [5500 / (SR / 2), 9000 / (SR / 2)], btype="bandpass", output="sos")
    hf = sosfiltfilt(sos, white_noise(1.0, SR, 1.0, seed=2).astype(np.float64), axis=0).astype(np.float32)
    band_peak = float(np.max(np.abs(hf)))
    # A bandpass may exceed its input peak by ringing, but must stay under 1.5x.
    ok = band_peak < 1.5
    print(f"  {'OK  ' if ok else 'FAIL'} {'full-amplitude HF band':<44} peak={band_peak:.4f} ({db(band_peak):+.2f} dBFS, limit 1.5)")
    failures += not ok

    print()
    if failures:
        print(f"{failures} clipping check(s) FAILED")
        return 1
    print("all clipping checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
