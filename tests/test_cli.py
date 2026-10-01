"""End-to-end tests for the command line.

These drive `main()` exactly as a user would, in-process, and assert on what
gets printed. The reason is specific: the CLI is a formatting layer over
analysis results, and formatting is where unmeasurable values turn into
crashes. Digital silence has no noise floor and no SNR, so any line that
prints one has to handle its absence instead of subtracting or formatting it.

The files written are checked for level and channel count rather than compared
byte-for-byte, because the WAV is deliberately quantised to 16-bit.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from aelf.cli import main
from conftest import SR, silence, speech_like, speech_plus_noise, white_noise, write_fixture


def test_enhance_writes_a_file_and_reports_measurements(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "noisy.wav"
    write_fixture(source, speech_plus_noise(seconds=2.0, noise_sigma=0.05, speech_gain=0.3))
    out = tmp_path / "clean.wav"

    assert main(["enhance", str(source), "-o", str(out), "--strength", "0.75"]) == 0

    printed = capsys.readouterr().out
    assert "Clean audio:" in printed
    assert "Your original is untouched:" in printed
    # Loudness is LUFS, not a peak level; labelling it dBFS would be wrong.
    assert "LUFS" in printed
    assert "loudness" in printed
    # Every stage that ran has to be named, including in the summary.
    assert "high-pass" in printed
    assert "noise suppression" in printed
    assert out.exists()
    data, sr = sf.read(str(out), always_2d=True)
    assert sr == SR
    assert data.shape == (int(SR * 2.0), 1)


def test_enhance_does_not_modify_the_source(tmp_path: Path) -> None:
    source = tmp_path / "noisy.wav"
    write_fixture(source, speech_plus_noise(seconds=1.0, noise_sigma=0.05, speech_gain=0.3))
    before = source.read_bytes()

    assert main(["enhance", str(source), "-o", str(tmp_path / "clean.wav")]) == 0
    assert source.read_bytes() == before


def test_enhance_reports_that_a_digitally_silent_file_has_no_measurable_floor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Silence used to crash the CLI with a None format error.

    The noise profile reports no floor and no SNR for a silent signal, so the
    report has to say so rather than print a number or subtract None.
    """
    source = tmp_path / "silent.wav"
    write_fixture(source, silence(seconds=1.0))
    out = tmp_path / "silent_clean.wav"

    assert main(["enhance", str(source), "-o", str(out)]) == 0

    printed = capsys.readouterr().out
    assert "n/a" in printed
    assert "SNR not measurable" in printed
    assert "could not be measured" in printed
    assert out.exists()


def test_enhance_accepts_stereo_and_keeps_both_channels(tmp_path: Path) -> None:
    left = speech_like(1.0, SR, 120.0, 3) * 0.3
    right = speech_like(1.0, SR, 210.0, 4) * 0.3
    stereo = np.stack([left, right], axis=1) + white_noise(1.0, SR, 0.03, seed=5)[:, None]
    source = tmp_path / "stereo.wav"
    write_fixture(source, stereo)
    out = tmp_path / "stereo_clean.wav"

    assert main(["enhance", str(source), "-o", str(out)]) == 0

    data, _ = sf.read(str(out), always_2d=True)
    assert data.shape[1] == 2


def test_zero_strength_turns_off_suppression_but_still_tidies(tmp_path: Path) -> None:
    """Two separate switches, so two separate behaviours to pin down.

    `--strength 0.0` means no noise removal. The tidying stages are a
    different switch, which is why the help text asks for --no-tidy to reach
    audio that is genuinely untouched.
    """
    source = tmp_path / "noisy.wav"
    write_fixture(source, speech_plus_noise(seconds=1.0, noise_sigma=0.05, speech_gain=0.3))
    suppressed_off = tmp_path / "suppression_off.wav"
    untouched = tmp_path / "untouched.wav"

    assert main(["enhance", str(source), "-o", str(suppressed_off), "--strength", "0.0"]) == 0
    assert main(["enhance", str(source), "-o", str(untouched), "--strength", "0.0", "--no-tidy"]) == 0

    original, _ = sf.read(str(source), always_2d=True)
    tidied, _ = sf.read(str(suppressed_off), always_2d=True)
    as_input, _ = sf.read(str(untouched), always_2d=True)

    # Tidying runs, so the filter stages do change the waveform.
    assert np.max(np.abs(original - tidied)) > 1e-3
    # With both switches off the audio is returned as it arrived.
    assert np.max(np.abs(original - as_input)) < 1e-3


def test_enhance_normalises_to_a_requested_target(tmp_path: Path) -> None:
    source = tmp_path / "quiet.wav"
    write_fixture(source, speech_like(2.0, SR, 120.0, 9) * 0.05)
    out = tmp_path / "loud.wav"

    assert main(["enhance", str(source), "-o", str(out), "--target", "-16"]) == 0

    from aelf.analysis.levels import measure_levels
    from aelf.io.decode import decode_to_canonical

    measured = measure_levels(decode_to_canonical(out))
    assert measured.lufs_integrated == pytest.approx(-16.0, abs=0.5)
    assert not measured.is_clipped


def test_missing_input_exits_with_an_error_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["enhance", str(tmp_path / "nope.wav")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_unsupported_suffix_is_refused_with_a_readable_message(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unsupported format must come back as a sentence, not a traceback."""
    # Written as a real WAV, then renamed, so the bytes are valid and only the
    # extension is wrong - which is what the suffix check actually rejects.
    staging = tmp_path / "recording.wav"
    write_fixture(staging, speech_plus_noise(seconds=1.0, noise_sigma=0.02, speech_gain=0.3))
    source = tmp_path / "recording.xyz"
    source.write_bytes(staging.read_bytes())

    code = main(["enhance", str(source)])

    assert code == 1
    err = capsys.readouterr().err
    assert err.strip()
    assert "Traceback" not in err


def test_compare_reports_a_score_against_a_matching_reference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    reference = tmp_path / "reference.wav"
    enhanced = tmp_path / "enhanced.wav"
    speech = speech_like(2.0, SR, 120.0, 6)
    write_fixture(reference, speech)
    write_fixture(enhanced, speech + white_noise(2.0, SR, 0.01, seed=8) * 0.5)

    assert main(["compare", str(enhanced), str(reference)]) == 0

    printed = capsys.readouterr().out
    assert "signal-to-distortion" in printed
    assert "scale-invariant SDR" in printed
    assert "alignment drift" in printed


def test_compare_refuses_mismatched_lengths_and_explains(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A confident number from misaligned audio would be worse than no number."""
    short = tmp_path / "short.wav"
    long = tmp_path / "long.wav"
    write_fixture(short, speech_like(2.0, SR, 120.0, 6))
    write_fixture(long, speech_like(3.0, SR, 120.0, 6))

    assert main(["compare", str(short), str(long)]) == 1
    printed = capsys.readouterr().out
    assert "signal-to-distortion" not in printed
    assert "reference" in printed.lower()


def test_compare_reports_a_missing_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    present = tmp_path / "present.wav"
    write_fixture(present, speech_like(1.0, SR, 120.0, 6))

    assert main(["compare", str(present), str(tmp_path / "absent.wav")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_parser_exposes_the_three_documented_commands() -> None:
    from aelf.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["enhance", "x.wav"]).command == "enhance"
    assert parser.parse_args(["compare", "a.wav", "b.wav"]).command == "compare"
    assert parser.parse_args(["serve"]).command == "serve"


def test_serve_pins_the_server_to_loopback_and_disables_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The local-only claim has to hold for the server, not just our code.

    Streamlit's defaults listen on every interface and send usage statistics.
    If those flags are ever dropped, this fails instead of quietly exposing
    someone else's audio on the local network.
    """
    import sys as real_sys

    captured: dict[str, object] = {}

    class FakeStreamlitCli:
        @staticmethod
        def main() -> int:
            captured["argv"] = list(real_sys.argv)
            return 0

    fake_module = types.ModuleType("streamlit.web.cli")
    fake_module.main = FakeStreamlitCli.main  # type: ignore[attr-defined]
    monkeypatch.setitem(real_sys.modules, "streamlit.web.cli", fake_module)
    monkeypatch.setattr(real_sys, "argv", ["aelf", "serve", "--port", "8599"])

    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--port", "8599"])
    assert excinfo.value.code == 0

    argv = captured["argv"]
    assert isinstance(argv, list)
    assert "127.0.0.1" in argv
    assert argv[argv.index("--server.address") + 1] == "127.0.0.1"
    assert argv[argv.index("--browser.gatherUsageStats") + 1] == "false"
    assert argv[argv.index("--server.port") + 1] == "8599"


def test_a_missing_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2
