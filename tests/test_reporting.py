"""Tests for how results are reported, not just for what they are.

Every assertion here is about the wording reaching the user. The numbers are
checked elsewhere; what matters here is that a number is never presented in a
way that makes it mean something other than what it measures.

The case that shaped these: normalising a recording to a loudness target shifts
the noise floor as an absolute level, because every sample is scaled. A report
that compared floors before and after would say the background got *worse* on a
file where it measurably improved. The gap between the floor and the voice does
not move with gain, so that is what the summary leans on when the level changed.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aelf.cli import main
from conftest import SR, speech_like, white_noise, write_fixture


class TestSilentInput:
    def test_enhance_on_silence_reports_nothing_measurable(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A silent file has no floor and no SNR. The report must say so."""
        source = tmp_path / "silent.wav"
        write_fixture(source, np.zeros(int(SR * 1.0), dtype=np.float32))
        out = tmp_path / "silent_clean.wav"

        assert main(["enhance", str(source), "-o", str(out)]) == 0

        printed = capsys.readouterr().out
        assert "n/a" in printed
        assert "could not be measured" in printed
        # The critical part: no invented figures.
        assert "dropped by" not in printed
        assert "improved by" not in printed

    def test_the_original_is_still_left_alone_for_a_silent_file(self, tmp_path: Path) -> None:
        source = tmp_path / "silent.wav"
        write_fixture(source, np.zeros(int(SR * 1.0), dtype=np.float32))
        before = source.read_bytes()

        assert main(["enhance", str(source), "-o", str(tmp_path / "out.wav")]) == 0
        assert source.read_bytes() == before


class TestLevelChangeReporting:
    def test_normalising_a_file_reports_the_snr_not_just_the_shifted_floor(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Raising the loudness raises the noise floor too. Saying only that the
        floor rose would be true and useless, so the SNR is reported alongside."""
        speech = speech_like(3.0, SR, 120.0, 42) * 0.3
        noisy = speech + white_noise(3.0, SR, 0.02, seed=7)
        source = tmp_path / "noisy.wav"
        write_fixture(source, noisy)
        out = tmp_path / "loud.wav"

        assert main(["enhance", str(source), "-o", str(out), "--strength", "0.0", "--target", "-16"]) == 0

        printed = capsys.readouterr().out
        assert "speech-to-noise" in printed
        # The floor line must still be present as an absolute level.
        assert "noise floor" in printed

    def test_a_turning_loudness_up_does_not_claim_the_background_got_worse(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The failure mode this exists to prevent.

        Normalising upward scales the whole file, so the absolute noise floor
        rises even while the background improves relative to the voice. With
        suppression genuinely running, the summary must not read the gain change
        as suppression having failed.
        """
        speech = speech_like(3.0, SR, 120.0, 42) * 0.03
        noisy = speech + white_noise(3.0, SR, 0.05, seed=7)
        source = tmp_path / "quiet_noisy.wav"
        write_fixture(source, noisy)
        out = tmp_path / "normalised.wav"

        assert main(["enhance", str(source), "-o", str(out), "--strength", "0.75", "--target", "-14"]) == 0

        printed = capsys.readouterr().out
        assert "barely moved" not in printed, "a gain change must not be read as suppression failing"

        # The direction of the floor movement has to match the numbers printed
        # directly above it. Normalising upward raises an absolute floor, so a
        # rising floor must be described as a rise; reporting the reduction
        # instead would invert the sign and quietly mislead.
        floor_line = next(
            line for line in printed.splitlines() if "->" in line and "dBFS" in line and "noise" in line
        )
        # Both levels appear as "<number> dBFS" on the line; take them in order.
        before, after = (float(value) for value in re.findall(r"(-?[\d.]+) dBFS", floor_line))
        moved = after - before
        assert "moved +" in printed, f"a floor that rose must be described as rising: {printed!r}"
        assert f"moved {moved:+.1f} dB" in printed, f"reported movement disagrees with {floor_line!r}"

    def test_a_silent_reference_is_refused_rather_than_scored(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Scoring against silence would produce a large confident number."""
        enhanced = tmp_path / "enhanced.wav"
        reference = tmp_path / "silent_reference.wav"
        write_fixture(enhanced, speech_like(1.0, SR, 120.0, 1))
        write_fixture(reference, np.zeros(int(SR * 1.0), dtype=np.float32))

        code = main(["compare", str(enhanced), str(reference)])

        assert code == 1
        printed = capsys.readouterr().out
        assert "dB" not in printed.split("Without one")[-1] or "reference" in printed.lower()


class TestUnits:
    def test_loudness_is_never_labelled_dbfs(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """LUFS and dBFS are different quantities. Conflating them is the sort
        of small inaccuracy that makes every other number suspect."""
        source = tmp_path / "speech.wav"
        write_fixture(source, speech_like(2.0, SR, 120.0, 3) * 0.2)

        assert main(["enhance", str(source), "-o", str(tmp_path / "out.wav")]) == 0

        printed = capsys.readouterr().out
        # The report body is indented; the unindented lines echo paths, which
        # pytest names after the test and would otherwise match on. Splitting
        # the "loudness ..., peak ..." header on its comma keeps the peak's own
        # dBFS label out of the loudness half, which is correct.
        for line in printed.splitlines():
            if not line.startswith("  "):
                continue
            for field in line.split(","):
                if "loudness" in field.lower():
                    assert "LUFS" in field
                    assert "dBFS" not in field, f"loudness mislabelled as a peak level: {field!r}"

    def test_peak_is_labelled_dbfs(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        source = tmp_path / "speech.wav"
        write_fixture(source, speech_like(2.0, SR, 120.0, 3) * 0.2)

        assert main(["enhance", str(source), "-o", str(tmp_path / "out.wav")]) == 0

        printed = capsys.readouterr().out
        peak_lines = [line for line in printed.splitlines() if line.startswith("  ") and "peak" in line.lower()]
        assert peak_lines
        assert any("dBFS" in line for line in peak_lines)

    def test_a_written_file_never_exceeds_the_ceiling(self, tmp_path: Path) -> None:
        """Whatever the report claims, the bytes on disk have to hold."""
        speech = speech_like(2.0, SR, 120.0, 3)
        loud = speech + white_noise(2.0, SR, 0.3, seed=2)
        source = tmp_path / "loud.wav"
        write_fixture(source, loud)
        out = tmp_path / "loud_clean.wav"

        assert main(["enhance", str(source), "-o", str(out), "--strength", "1.0", "--target", "-9"]) == 0

        data, _ = sf.read(str(out), always_2d=True)
        assert float(np.max(np.abs(data))) < 1.0
