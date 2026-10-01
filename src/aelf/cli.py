"""Command-line interface: drop a file in, get a cleaner file out.

Usage
    python -m aelf.cli enhance INPUT [-o OUTPUT] [--strength 0.5]
    python -m aelf.cli compare ENHANCED REFERENCE
    python -m aelf.cli serve

Examples
    python -m aelf.cli enhance noisy.wav
    python -m aelf.cli enhance noisy.wav -o clean.wav --strength 0.8
    python -m aelf.cli enhance noisy.mp3 --target -16

With no -o, the result lands in work/outputs/ next to a short report.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from pathlib import Path

from .analysis.levels import measure_levels
from .analysis.noise_profile import estimate_noise_profile
from .config import PATHS, TARGET_LUFS_SPEECH
from .errors import AelfError, MetricUnavailableError
from .io.decode import check_supported, decode_to_canonical
from .io.encode import write_wav
from .io.integrity import IntegrityGuard
from .pipeline import process


def _fmt_db(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f} dBFS"


def _fmt_lufs(value: float | None) -> str:
    """Loudness in LUFS, which is not a peak level and must not be labelled dBFS."""
    return "n/a" if value is None else f"{value:.1f} LUFS"


def _fmt_snr(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f} dB"


def cmd_enhance(args: argparse.Namespace) -> int:
    source = Path(args.input).expanduser()
    if not source.exists():
        print(f"error: no such file: {source}", file=sys.stderr)
        return 2

    print(f"Reading {source} ...")
    check_supported(source)
    with IntegrityGuard([source]) as guard:
        buffer = decode_to_canonical(source)

        print(
            f"  {buffer.duration_s:.2f}s, {buffer.channels} channel(s), "
            f"{buffer.sample_rate:,} Hz, {source.stat().st_size / 1024:.0f} KB"
        )

        before = measure_levels(buffer)
        noise_before = estimate_noise_profile(buffer)
        print(f"  loudness {_fmt_lufs(before.lufs_integrated)}, peak {_fmt_db(before.peak_dbfs)}")
        snr_text = (
            "SNR not measurable"
            if noise_before.estimated_snr_db is None
            else f"SNR about {noise_before.estimated_snr_db:.1f} dB"
        )
        print(f"  estimated noise floor {_fmt_db(noise_before.noise_floor_dbfs)}, {snr_text}")
        if before.is_clipped:
            print(
                f"  warning: {before.clipped_sample_count} samples already at or over full scale; "
                "those cannot be recovered"
            )

        print(f"Processing (strength {args.strength:.2f}) ...")
        result = process(buffer, strength=args.strength, tidy=not args.no_tidy, target_lufs=args.target)
        print(f"  done in {result.processing_time_s:.2f}s using {result.backend}")
        for line in result.applied:
            print(f"  - {line}")
        for line in result.notes:
            print(f"  note: {line}")
        output = result.audio

        PATHS.ensure()
        destination = Path(args.output).expanduser() if args.output else PATHS.outputs / f"{source.stem}_clean.wav"
        written = write_wav(output, destination, guard=guard)

        print()
        print(f"Clean audio: {written}  ({written.stat().st_size / (1024 * 1024):.1f} MB)")
        print(f"Your original is untouched: {source}")

        after = measure_levels(output)
        floor_after = estimate_noise_profile(output)
        print()
        print("What changed")
        print(f"  noise floor  {_fmt_db(noise_before.noise_floor_dbfs)} -> {_fmt_db(floor_after.noise_floor_dbfs)}")
        print(f"  speech-to-noise  {_fmt_snr(noise_before.estimated_snr_db)} -> {_fmt_snr(floor_after.estimated_snr_db)}")
        print(f"  loudness     {_fmt_lufs(before.lufs_integrated)} -> {_fmt_lufs(after.lufs_integrated)}")
        print(f"  peak         {_fmt_db(before.peak_dbfs)} -> {_fmt_db(after.peak_dbfs)}")

        # Either value can be None for digital silence, where there is nothing
        # to measure; subtracting None would raise rather than report.
        reduction = None
        if noise_before.noise_floor_dbfs is not None and floor_after.noise_floor_dbfs is not None:
            reduction = noise_before.noise_floor_dbfs - floor_after.noise_floor_dbfs
        # The noise floor is an absolute level, so normalising the recording
        # shifts it even when the noise is genuinely quieter. The gap between
        # the floor and the programme does not, which is what actually answers
        # "did the background get better".
        snr_gain = None
        if noise_before.estimated_snr_db is not None and floor_after.estimated_snr_db is not None:
            snr_gain = floor_after.estimated_snr_db - noise_before.estimated_snr_db
        print()
        print("What this does and does not tell you")
        if reduction is None:
            print("  A noise floor could not be measured, so there is no figure to report here.")
        elif snr_gain is not None and snr_gain > 1.0 and reduction < 1.0:
            # The floor rose because the whole recording got louder, not
            # because the noise got worse. Say which, or the number misleads.
            # Signed as after-minus-before, so a rise reads as a rise.
            moved = -(reduction)
            print(
                f"  The measured noise floor moved {moved:+.1f} dB, but that mostly reflects the "
                f"loudness change above. Relative to the voice, the background improved by {snr_gain:.1f} dB."
            )
        elif reduction > 1.0:
            print(f"  The measured noise floor dropped by {reduction:.1f} dB.")
        else:
            print(
                "  The measured noise floor barely moved. This recording may not have much steady "
                "background noise for this method to remove."
            )
        print("  That is a measurement of the noise floor, not a score for how good the")
        print("  recording sounds. To measure improvement properly you need the original")
        print("  clean recording to compare against. Run 'compare' if you have one.")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare a processed file against a clean reference, if one exists."""
    enhanced_path = Path(args.enhanced).expanduser()
    reference_path = Path(args.reference).expanduser()

    if not enhanced_path.exists():
        print(f"error: no such file: {enhanced_path}", file=sys.stderr)
        return 2
    if not reference_path.exists():
        print(f"error: no such file: {reference_path}", file=sys.stderr)
        return 2

    enhanced = decode_to_canonical(enhanced_path)
    reference = decode_to_canonical(reference_path)

    from .evaluate.compare import compare_to_reference, describe_missing_reference

    try:
        report = compare_to_reference(enhanced, reference)
    except MetricUnavailableError as exc:
        print(exc.user_message)
        print()
        print(describe_missing_reference())
        print()
        print(f"  {exc.detail}")
        return 1

    print(f"Comparing {enhanced_path.name} against {reference_path.name}")
    print()
    print(f"  alignment drift      {report.duration_drift_s * 1000:.1f} ms")
    print(f"  signal-to-distortion {report.sdr_db:+.2f} dB")
    print(f"  scale-invariant SDR  {report.sisdr_db:+.2f} dB")
    print(f"  confidence           {report.confidence.label}")
    print()
    print(f"  {report.verdict}")
    for note in report.notes:
        print(f"  note: {note}")
    if not report.notes:
        print()
        print("These scores require a clean recording of the same speech, made the same way.")
        print("Without one, only noise-floor and level measurements are available.")
    return 0


def _open_browser_when_ready(url: str, port: int, timeout_s: float = 60.0) -> threading.Thread:
    """Open the browser once the server actually answers, then not before.

    A fixed delay is a guess. Streamlit binds in about two seconds on a warm
    start and can take fifteen or more on the first, when it is unpacking
    configuration and building caches. Opening on a timer therefore lands the
    browser on a port nothing is listening on, which the user sees as
    "connection refused" and reasonably reads as a broken app.

    Polling for a real answer removes the guess. The thread is returned so a
    caller can wait on it; the daemon flag means a browser that never opens
    cannot keep the server alive.
    """
    import socket
    import time

    def wait() -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.25)

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    return thread


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the browser interface.

    The server is pinned to loopback and Streamlit's usage statistics are
    switched off explicitly, rather than relying on a config file being found.
    A privacy claim that depends on configuration nobody has to notice is not
    much of a claim: with the defaults, Streamlit binds every interface and
    phones home.
    """
    url = f"http://127.0.0.1:{args.port}"
    print(f"Starting AELF at {url}")
    print("The page opens in your browser as soon as it is ready.")
    print("Press Ctrl+C to stop.")
    try:
        from streamlit.web import cli as st_cli
    except ImportError:
        print("error: Streamlit is not installed. Run: pip install -e '.[ui]'", file=sys.stderr)
        return 2
    app = Path(__file__).resolve().parents[2] / "app.py"
    if not app.exists():
        print(f"error: could not find the interface at {app}", file=sys.stderr)
        return 2
    sys.argv = [
        "streamlit",
        "run",
        str(app),
        "--server.port",
        str(args.port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    _open_browser_when_ready(url, args.port)
    sys.exit(st_cli.main())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aelf",
        description="Local-only audio enhancement. Nothing leaves your machine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    enhance_cmd = sub.add_parser("enhance", help="Clean up a recording")
    enhance_cmd.add_argument("input", help="WAV, MP3 or M4A file")
    enhance_cmd.add_argument("-o", "--output", help="Where to write the result (default: work/outputs/)")
    enhance_cmd.add_argument(
        "--strength",
        type=float,
        default=0.5,
        help=(
            "How much background noise to remove, 0.0 to 1.0 (default: 0.5). "
            "0.0 turns off noise suppression only; add --no-tidy to leave the audio completely alone"
        ),
    )
    enhance_cmd.add_argument(
        "--target",
        type=float,
        default=None,
        metavar="LUFS",
        help=f"Even out the loudness to this level (default: none. {TARGET_LUFS_SPEECH:.0f} suits spoken word)",
    )
    enhance_cmd.add_argument(
        "--no-tidy",
        action="store_true",
        help="Skip the rumble removal, speech EQ and harsh-sound control, leaving the voice as enhanced",
    )
    enhance_cmd.set_defaults(func=cmd_enhance)

    compare_cmd = sub.add_parser("compare", help="Score output against a clean reference")
    compare_cmd.add_argument("enhanced", help="The processed file")
    compare_cmd.add_argument("reference", help="The clean original recording of the same speech")
    compare_cmd.set_defaults(func=cmd_compare)

    serve_cmd = sub.add_parser("serve", help="Open the browser interface")
    serve_cmd.add_argument("--port", type=int, default=8501)
    serve_cmd.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except AelfError as exc:
        print(f"\n{exc.user_message}", file=sys.stderr)
        if exc.remedy:
            print(f"  {exc.remedy}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
