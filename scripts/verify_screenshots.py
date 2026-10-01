"""Verify every screenshot command in the report actually works.

The report tells the reader to capture ten screenshots. Each one names a
command or a view. A report that instructs a command that no longer runs is
worse than no report, so this checks each of them before the reader tries.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "Scripts" / "aelf.exe"

# (label, argv, substrings that must appear in stdout)
CHECKS = [
    ("2  aelf --help", ["--help"],
     ["enhance", "analyze", "compare", "serve", "Challenge 3"]),
    ("3  analyze submitted MP3", ["analyze", "SUBMISSION/v4 test (final).mp3"],
     ["duration", "285.0s", "dual mono", "noise floor", "-81.49",
      "SNR", "59.9", "stationary", "66.7 Hz", "non-stationary"]),
    ("5  strength sweep", ["analyze", "work/uploads/war-intercept.wav", "--sweep"],
     ["strength sweep", "0.00", "0.25", "0.50", "0.75", "1.00",
      "9.86", "9.62", "removing speech"]),
    ("6  enhance", ["enhance", "work/uploads/war-intercept.wav",
                    "--strength", "0.75", "--target", "-30",
                    "-o", "work/outputs/verify_shot6.wav"],
     ["What changed", "noise floor", "speech-to-noise", "loudness",
      "peak", "does and does not"]),
    # Matched pair. The demo files share a duration on purpose; the mismatched
    # sample pair exists to prove the alignment guard fires.
    ("7  compare (matched)", ["compare", "work/outputs/demo_v3.wav",
                              "work/uploads/demo_clean_reference.wav"],
     ["signal-to-distortion", "scale-invariant SDR", "confidence"]),
    # Not -q: quiet mode prints dots and drops the "N passed" summary line, so
    # the screenshot would be missing the very figure the report cites.
    ("9  pytest", ["-m", "pytest"],
     ["passed"]),
]

failures = []
for label, argv, needles in CHECKS:
    exe = PY if argv[0] != "-m" else str(REPO / ".venv" / "Scripts" / "python.exe")
    result = subprocess.run([exe, *argv], cwd=REPO, capture_output=True,
                            text=True)
    out = result.stdout + result.stderr
    # Case-insensitive: the CLI labels rows in lower case and this check should
    # not fail on a capital letter.
    lowered = out.lower()
    missing = [n for n in needles if n.lower() not in lowered]
    if result.returncode != 0 or missing:
        failures.append((label, result.returncode, missing, out[-500:]))
        print(f"FAIL {label}  rc={result.returncode}  missing={missing}")
    else:
        print(f"ok   {label}")

print()
if failures:
    print(f"{len(failures)} screenshot command(s) would fail.")
    sys.exit(1)
print(f"all {len(CHECKS)} screenshot commands reproduce their documented output.")
