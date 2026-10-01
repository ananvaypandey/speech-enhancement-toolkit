"""Browser interface.

Deliberately plain: upload a file, choose a strength, download the result.
Nothing is uploaded anywhere. Streamlit runs this locally and the audio is
passed in memory, never off the machine.

The ordering of the page is the ordering of the work: add audio, adjust,
process, download. Every number shown is measured from the audio rather than
estimated, and where a number would need a reference that does not exist, the
page says so instead of showing one.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

from aelf.analysis.levels import measure_levels
from aelf.analysis.noise_profile import estimate_noise_profile
from aelf.config import (
    CLIP_CEILING_DBFS,
    PATHS,
    SUPPORTED_INPUT_SUFFIXES,
    TARGET_LUFS_LISTENING,
    TARGET_LUFS_SPEECH,
)
from aelf.enhance import available_backends
from aelf.errors import AelfError
from aelf.io.decode import load
from aelf.io.encode import true_peak_limit, write_wav
from aelf.io.integrity import IntegrityGuard
from aelf.pipeline import process
from aelf.types import AudioBuffer

st.set_page_config(page_title="AELF — local audio cleanup", page_icon="graphic_eq", layout="wide")

STRENGTH_HELP = {
    0.0: "Off. No noise removal at all. The tidying steps still apply unless you turn those off too.",
    0.25: "Light. Takes the edge off steady hiss.",
    0.5: "Recommended. Clearly quieter background noise, speech still natural.",
    0.75: "Strong. Removes more, at the cost of a little speech brightness.",
    1.0: "Maximum. Removes the most this method can without hollowing out speech.",
}


def _fmt_db(value: float | None) -> str:
    return "not measurable" if value is None else f"{value:+.1f} dBFS"


def _fmt_lufs(value: float | None) -> str:
    """Loudness in LUFS. Not a peak level, so it must not carry a dBFS label."""
    return "not measurable" if value is None else f"{value:.1f} LUFS"


def _db_arrow(before: float | None, after: float | None, lower_is_better: bool = False) -> str:
    if before is None or after is None:
        return "not measurable"
    delta = after - before
    if abs(delta) < 0.05:
        return "unchanged"
    improved = delta < 0 if lower_is_better else delta > 0
    arrow = "down" if delta < 0 else "up"
    return f"{arrow} {abs(delta):.1f} dB ({'better' if improved else 'worse'})"


@st.cache_data(show_spinner=False)
def _decode_cached(raw: bytes, name: str) -> tuple[AudioBuffer, float]:
    """Decode once per (file, settings). Streamlit re-runs the whole script on
    every interaction, so without this the same file is decoded repeatedly."""
    PATHS.ensure()
    # Written to the uploads directory so the decoder has a real path with a
    # real suffix; MP3 and M4A need the extension to pick a codec.
    staged = PATHS.uploads / name
    staged.write_bytes(raw)
    started = time.perf_counter()
    buffer = load(staged)
    return buffer, time.perf_counter() - started


@st.cache_data(show_spinner=False)
def _process_cached(
    samples: np.ndarray,
    sample_rate: int,
    strength: float,
    tidy: bool,
    target_lufs: float | None,
) -> tuple[np.ndarray, str, float, list[str], list[str]]:
    result = process(
        AudioBuffer(samples=samples, sample_rate=sample_rate),
        strength=strength,
        tidy=tidy,
        target_lufs=target_lufs,
    )
    return result.audio.samples, result.backend, result.processing_time_s, result.applied, result.notes


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    """Encode float samples as WAV bytes for the browser player.

    `st.audio` accepts a numpy array, but its internal writer uses the stdlib
    `wave` module and chokes on anything but 1-D int16 or float32 — a stereo
    buffer raises `struct.error: argument out of range`. Encoding here with
    soundfile, which already handles the channel layout, avoids that and keeps
    the player and the download byte-identical.
    """
    buf = io.BytesIO()
    sf.write(buf, np.asarray(samples, dtype=np.float32), sample_rate, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _levels_for(samples: np.ndarray, sample_rate: int) -> object:
    return measure_levels(AudioBuffer(samples=samples, sample_rate=sample_rate))


def _noise_for(samples: np.ndarray, sample_rate: int) -> object:
    return estimate_noise_profile(AudioBuffer(samples=samples, sample_rate=sample_rate))


st.title("AELF — local audio cleanup")
st.caption(
    "Runs entirely on this computer. Your audio is not uploaded anywhere, and your original file is never modified."
)

with st.sidebar:
    st.header("Settings")
    st.markdown(
        "**Method.** "
        + ", ".join(available_backends())
        + ". Needs nothing downloaded and works on speech, hiss, hum and fan whir. "
        "It cannot follow noise that changes over time the way a trained model can."
    )
    strength = st.slider(
        "How much noise to remove",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.25,
        help=STRENGTH_HELP[0.5],
    )
    st.caption(STRENGTH_HELP.get(strength, ""))
    normalise = st.checkbox("Even out the loudness", value=False)
    target = st.slider(
        "Target loudness (LUFS)",
        min_value=-30.0,
        max_value=-9.0,
        value=TARGET_LUFS_SPEECH,
        step=1.0,
        disabled=not normalise,
        help=(
            f"{TARGET_LUFS_SPEECH:.0f} LUFS is the broadcast standard for spoken word. "
            f"{TARGET_LUFS_LISTENING:.0f} LUFS suits headphones."
        ),
    )
    tidy = st.checkbox(
        "Tidy the voice",
        value=True,
        help=(
            "Removes low rumble, lifts the frequencies speech is understood on, and takes the edge "
            "off harsh sssss sounds. Turn this off to leave the voice exactly as enhanced."
        ),
    )

uploaded = st.file_uploader(
    "Add your audio file",
    type=sorted(s.lstrip(".") for s in SUPPORTED_INPUT_SUFFIXES),
    help="WAV, MP3, M4A, FLAC, OGG or Opus.",
)

if uploaded is None:
    st.info("Add a file above to begin. There is nothing else to set up.")
    st.stop()

raw = uploaded.getvalue()
try:
    with st.spinner("Reading your audio ..."):
        buffer, _ = _decode_cached(raw, uploaded.name)
except AelfError as exc:
    st.error(exc.user_message)
    st.caption(exc.remedy)
    st.stop()
except Exception as exc:
    st.error("The file could not be read.")
    st.caption(f"{type(exc).__name__}: {exc}")
    st.stop()

before_levels = _levels_for(buffer.samples, buffer.sample_rate)
before_noise = _noise_for(buffer.samples, buffer.sample_rate)

left, right = st.columns(2)
with left:
    st.subheader("Your audio")
    st.write(f"**{uploaded.name}**")
    st.write(f"{buffer.duration_s:.1f} seconds · {buffer.channels} channel(s) · {buffer.sample_rate:,} Hz")
    st.write(f"Loudness {_fmt_lufs(before_levels.lufs_integrated)} · peak {_fmt_db(before_levels.peak_dbfs)}")
    st.write(f"Estimated noise floor {_fmt_db(before_noise.noise_floor_dbfs)}")
    st.audio(_wav_bytes(buffer.samples, buffer.sample_rate), format="audio/wav")

if before_levels.is_clipped:
    st.warning(
        f"{before_levels.clipped_sample_count} samples in your file are already at or over full scale. "
        "That distortion is baked in and cannot be undone, so cleanup will be limited by it."
    )
if not before_noise.is_stationary:
    st.info(
        "This recording does not look like it has steady background noise. "
        "Cleaning it up may not help much — try it and listen."
    )

with right:
    st.subheader("After cleanup")
    with st.spinner("Cleaning up ..."):
        processed, backend, elapsed, applied, notes = _process_cached(
            buffer.samples,
            buffer.sample_rate,
            strength,
            tidy,
            target if normalise else None,
        )

    st.audio(_wav_bytes(processed, buffer.sample_rate), format="audio/wav")
    st.caption(f"Processed in {elapsed:.2f}s using {backend}.")

    after_levels = _levels_for(processed, buffer.sample_rate)
    after_noise = _noise_for(processed, buffer.sample_rate)

    st.markdown("**What changed**")
    st.write(
        f"- Noise floor: {_fmt_db(before_noise.noise_floor_dbfs)} → {_fmt_db(after_noise.noise_floor_dbfs)} "
        f"({_db_arrow(before_noise.noise_floor_dbfs, after_noise.noise_floor_dbfs, lower_is_better=True)})"
    )
    st.write(
        f"- Loudness: {_fmt_lufs(before_levels.lufs_integrated)} → {_fmt_lufs(after_levels.lufs_integrated)}"
    )
    st.write(f"- Peak: {_fmt_db(before_levels.peak_dbfs)} → {_fmt_db(after_levels.peak_dbfs)}")
    if after_levels.is_clipped:
        st.write(f"- Samples at or over full scale: {after_levels.clipped_sample_count}")

    st.markdown("**What was done**")
    for line in applied:
        st.write(f"- {line}")

    st.markdown("**How to read this**")
    # Digital silence has no measurable floor, so the difference may not exist
    # at all. Say that rather than printing a number derived from None.
    reduction = None
    if before_noise.noise_floor_dbfs is not None and after_noise.noise_floor_dbfs is not None:
        reduction = before_noise.noise_floor_dbfs - after_noise.noise_floor_dbfs
    if reduction is None:
        st.write("A noise floor could not be measured for this file, so there is no figure to report.")
    elif reduction > 1.0:
        st.write(f"The measured background noise came down by {reduction:.1f} dB.")
    else:
        st.write(
            "The measured background noise barely moved. Either this recording has very little steady "
            "noise, or the noise is changing over time and this method cannot follow it."
        )
    st.write(
        "These are measurements of the audio, not a score for how it sounds. To measure improvement "
        "properly you need the original clean recording of the same speech, made the same way — "
        "run `aelf compare` with that file to get a real score."
    )
    for note in notes:
        st.caption(note)

with st.expander("Audio before and after, back to back", expanded=False):
    st.caption("Useful for hearing the difference directly. Listen for speech clarity, not just hiss.")
    gap = np.zeros(int(buffer.sample_rate * 0.5), dtype=np.float32)
    combo = np.concatenate([buffer.mono(), gap, AudioBuffer(samples=processed, sample_rate=buffer.sample_rate).mono()])
    st.audio(_wav_bytes(combo, buffer.sample_rate), format="audio/wav")

st.subheader("Download")

PATHS.ensure()
safe_stem = Path(uploaded.name).stem
default_name = f"{safe_stem}_clean.wav"

wav_data = _wav_bytes(true_peak_limit(processed), buffer.sample_rate)

col1, col2 = st.columns([1, 2])
with col1:
    st.download_button(
        "Download cleaned audio",
        data=wav_data,
        file_name=default_name,
        mime="audio/wav",
        type="primary",
    )
with col2:
    st.caption(
        f"16-bit WAV, with peaks held under {CLIP_CEILING_DBFS:.0f} dBFS so nothing distorts. "
        "Downloading does not write anywhere on disk; use the button below to keep a copy."
    )

if st.button("Also save to the outputs folder"):
    guard = IntegrityGuard([PATHS.uploads / uploaded.name]) if (PATHS.uploads / uploaded.name).exists() else None
    path = write_wav(
        AudioBuffer(samples=processed, sample_rate=buffer.sample_rate),
        PATHS.outputs / default_name,
        guard=guard,
    )
    st.success(f"Saved to {path}")

st.divider()
st.caption(
    "AELF — local audio cleanup. "
    "No telemetry, no uploads, no accounts. "
    "Cleanup removes steady background noise; it does not separate speakers, transcribe speech, or repair clipping."
)
