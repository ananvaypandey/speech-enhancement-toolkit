"""Tests for the decode/encode layer and the source-integrity guarantee.

The central test here is `test_full_pipeline_never_touches_source`: it runs
representative work over a real file on disk and asserts the input is
byte-identical afterwards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aelf.config import CLIP_CEILING_DBFS
from aelf.errors import (
    EmptyAudioError,
    SourceIntegrityError,
    SourceMutatedError,
    UnsupportedFormatError,
)
from aelf.io.decode import check_supported, decode_to_canonical, ffmpeg_path, load, resample
from aelf.io.encode import count_clipped, normalise, to_int16, true_peak_limit, write_wav
from aelf.io.integrity import IntegrityGuard, fingerprint, sha256_file
from aelf.types import AudioBuffer
from conftest import SR, clipped, silence, speech_plus_noise, tone, two_speakers, write_fixture


class TestIntegrityGuard:
    def test_fingerprint_is_content_addressed(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "a.wav", tone()))
        first = fingerprint(path)
        assert first.sha256 == sha256_file(path)
        assert first.size_bytes == path.stat().st_size

    def test_guard_detects_mutation(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "a.wav", tone()))
        guard = IntegrityGuard([path])
        sf.write(str(path), tone(880.0), SR, subtype="PCM_16")
        with pytest.raises(SourceMutatedError):
            guard.verify_unchanged()

    def test_guard_allows_unchanged(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "a.wav", tone()))
        guard = IntegrityGuard([path])
        guard.verify_unchanged()
        assert all(r["verified_unchanged"] for r in guard.summary())

    def test_write_over_source_is_blocked(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "src.wav", tone()))
        guard = IntegrityGuard([path])
        with pytest.raises(SourceIntegrityError):
            guard.assert_targets_safe([path])
        # And the encoder must refuse too, not just the guard.
        with pytest.raises(SourceIntegrityError):
            write_wav(AudioBuffer(samples=tone(), sample_rate=SR), path, guard=guard)

    def test_traversal_alias_is_blocked(self, tmp_path: Path) -> None:
        source = Path(write_fixture(tmp_path / "src.wav", tone()))
        alias = tmp_path / "sub" / ".." / "src.wav"
        (tmp_path / "sub").mkdir(exist_ok=True)
        guard = IntegrityGuard([source])
        with pytest.raises(SourceIntegrityError):
            guard.assert_targets_safe([alias])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            fingerprint(tmp_path / "nope.wav")

    def test_full_pipeline_never_touches_source(self, tmp_path: Path) -> None:
        """Simulate the real pipeline: read, process, write, verify."""
        source = Path(write_fixture(tmp_path / "recording.wav", speech_plus_noise()))
        before_hash = sha256_file(source)
        before_stat = source.stat()

        guard = IntegrityGuard([source])
        buffer = decode_to_canonical(source)

        from aelf.analysis.levels import measure_levels
        from aelf.postprocess.normalize import normalise_lufs

        measure_levels(buffer)
        processed = normalise_lufs(buffer, target_lufs=-23.0)

        out = tmp_path / "out" / "enhanced.wav"
        write_wav(processed, out, guard=guard)
        guard.verify_unchanged()

        assert sha256_file(source) == before_hash
        assert source.stat().st_size == before_stat.st_size
        assert source.stat().st_mtime_ns == before_stat.st_mtime_ns
        assert out.exists() and out.resolve() != source.resolve()


class TestDecode:
    def test_reads_pcm16_with_correct_amplitude(self, tmp_path: Path) -> None:
        # A 0.5-amplitude sine has an RMS of 0.5/sqrt(2) = 0.354. Stored as
        # 16-bit it must read back at that level and NOT be renormalised up
        # toward full scale.
        path = Path(write_fixture(tmp_path / "t.wav", tone(1000.0, 0.5, SR, amplitude=0.5)))
        buffer = load(path)
        assert buffer.sample_rate == SR
        assert buffer.channels == 1
        assert float(np.sqrt(np.mean(buffer.samples**2))) == pytest.approx(0.5 / np.sqrt(2.0), rel=1e-3)
        assert float(np.max(np.abs(buffer.samples))) == pytest.approx(0.5, rel=1e-2)

    def test_reads_stereo(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "s.wav", np.stack([tone(440.0), tone(660.0)], axis=1)))
        buffer = load(path)
        assert buffer.channels == 2
        assert buffer.duration_s == pytest.approx(1.0, abs=1e-3)

    def test_rejects_unsupported_suffix(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("not audio")
        with pytest.raises(UnsupportedFormatError):
            check_supported(path)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.wav"
        sf.write(str(path), np.zeros((0, 1), dtype=np.float32), SR, subtype="PCM_16")
        with pytest.raises(EmptyAudioError):
            load(path)

    def test_too_short_raises(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "tiny.wav", tone(1000.0, 0.005)))
        with pytest.raises(EmptyAudioError):
            load(path)

    def test_resample_preserves_duration(self) -> None:
        buffer = AudioBuffer(samples=tone(440.0, 1.0, 44_100), sample_rate=44_100)
        out = resample(buffer, 48_000)
        assert out.sample_rate == 48_000
        assert out.duration_s == pytest.approx(1.0, abs=0.01)

    def test_resample_is_noop_at_same_rate(self) -> None:
        buffer = AudioBuffer(samples=tone(), sample_rate=SR)
        assert resample(buffer, SR) is buffer

    def test_canonical_sr_is_48000(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "a.wav", tone(440.0, 0.5, 44_100), sr=44_100))
        assert decode_to_canonical(path).sample_rate == 48_000

    def test_decode_does_not_modify_input(self, tmp_path: Path) -> None:
        path = Path(write_fixture(tmp_path / "a.wav", tone()))
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        decode_to_canonical(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before

    def test_ffmpeg_available_for_compressed_formats(self) -> None:
        """MP3/M4A decode depends on ffmpeg; report rather than assume."""
        assert ffmpeg_path() is not None, "ffmpeg not found; compressed-format decoding will be unavailable"


class TestEncode:
    def test_true_peak_limit_enforces_ceiling(self) -> None:
        loud = (np.array([1.0, -1.0, 0.5, -0.25], dtype=np.float32) * 0.99)
        out = true_peak_limit(loud)
        ceiling = 10.0 ** (CLIP_CEILING_DBFS / 20.0)
        assert float(np.max(np.abs(out))) <= ceiling + 1e-6
        assert count_clipped(out) == 0

    def test_true_peak_limit_leaves_quiet_signal_alone(self) -> None:
        quiet = np.full(100, 0.1, dtype=np.float32)
        assert np.array_equal(true_peak_limit(quiet), quiet)

    def test_true_peak_limit_preserves_relative_shape(self) -> None:
        """Limiting is a uniform gain, so the waveform's proportions survive."""
        data = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        out = true_peak_limit(data)
        ratio_out = out[1:] / out[:-1]
        ratio_in = data[1:] / data[:-1]
        assert np.allclose(ratio_out, ratio_in, atol=1e-5)

    def test_true_peak_limit_handles_silence(self) -> None:
        assert not np.any(true_peak_limit(np.zeros(10, dtype=np.float32)))

    def test_written_file_has_no_clipped_samples(self, tmp_path: Path) -> None:
        target = tmp_path / "loud.wav"
        write_wav(AudioBuffer(samples=clipped(), sample_rate=SR), target)
        data, sr = sf.read(str(target), dtype="float32", always_2d=True)
        assert sr == SR
        assert count_clipped(data) == 0

    def test_roundtrip_preserves_samples(self, tmp_path: Path) -> None:
        original = tone(1000.0, 0.2, amplitude=0.4)
        target = tmp_path / "rt.wav"
        write_wav(AudioBuffer(samples=original, sample_rate=SR), target, subtype="FLOAT", limit=False)
        read_back, _ = sf.read(str(target), dtype="float32")
        assert np.allclose(read_back, original, atol=1e-6)

    def test_empty_buffer_refuses_to_write(self, tmp_path: Path) -> None:
        with pytest.raises(EmptyAudioError):
            write_wav(AudioBuffer(samples=np.zeros((0, 1), dtype=np.float32), sample_rate=SR), tmp_path / "e.wav")

    def test_int16_conversion_clips_hard(self) -> None:
        out = to_int16(np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32))
        assert out.tolist() == [-32767, -32767, 0, 32767, 32767]

    def test_normalise_reaches_target_peak(self) -> None:
        out = normalise(tone(1000.0, 0.1, amplitude=0.05), target_peak_dbfs=-3.0)
        assert float(np.max(np.abs(out))) == pytest.approx(10 ** (-3.0 / 20.0), rel=1e-4)

    def test_normalise_never_clips(self) -> None:
        out = normalise(speech_plus_noise() * 100.0, target_peak_dbfs=0.0)
        assert count_clipped(out) == 0


class TestStereoAndMultiChannel:
    def test_mono_downmix(self) -> None:
        buf = AudioBuffer(samples=np.stack([tone(440.0, 0.5, SR, 0.4), tone(440.0, 0.5, SR, 0.4)], axis=1), sample_rate=SR)
        assert buf.mono().shape == (24_000,)

    def test_replace_mono_duplicates_channels(self) -> None:
        buf = AudioBuffer(
            samples=np.stack([tone(440.0, 0.2, SR, 0.4), tone(660.0, 0.2, SR, 0.3)], axis=1), sample_rate=SR
        )
        out = buf.replace_mono(buf.mono())
        assert out.channels == 2
        assert np.allclose(out.samples[:, 0], out.samples[:, 1])

    def test_buffer_normalises_1d_to_2d(self) -> None:
        buf = AudioBuffer(samples=tone(440.0, 0.1), sample_rate=SR)
        assert buf.samples.ndim == 2
        assert buf.channels == 1

    def test_buffer_casts_to_float32(self) -> None:
        buf = AudioBuffer(samples=np.zeros(1000, dtype=np.float64), sample_rate=SR)
        assert buf.samples.dtype == np.float32

    def test_two_speaker_fixture_is_mono_float32(self) -> None:
        data = two_speakers()
        assert data.dtype == np.float32 and data.ndim == 1

    def test_silence_detected(self) -> None:
        assert AudioBuffer(samples=silence(), sample_rate=SR).is_silent
        assert not AudioBuffer(samples=tone(), sample_rate=SR).is_silent
