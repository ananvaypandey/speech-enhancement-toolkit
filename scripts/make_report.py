"""Generate SUBMISSION/report.docx.

python-docx has no markdown parser, so the document is built with explicit
calls. Every measured number here came from running the code; nothing is
rounded in a flattering direction.

Run from the repo root:
    python scripts/make_report.py
"""
from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "SUBMISSION" / "report.docx"

NAME = "Ananvay Pandey"
EMAIL = "ananvaypandey29@gmail.com"
INSTITUTE = "Pranveer Singh Institute of Technology"
ENROLMENT = "2401640100170"
STUDENT_ID = "STU6a1db459939a31780331609"
PROGRAMME = "B.Tech Computer Science and Engineering"


def heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def para(doc: Document, text: str, *, bold: bool = False, italic: bool = False) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    p.paragraph_format.space_after = Pt(6)


def bullet(doc: Document, text: str, level: int = 0) -> None:
    doc.add_paragraph(text, style="List Bullet" if level == 0 else "List Bullet 2")


def numbered(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Number")


def code(doc: Document, text: str) -> None:
    """A terminal-looking block. Screenshots are placeholders; this is the
    verifiable text alongside them."""
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = "Consolas"
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    pf = p.paragraph_format
    pf.left_indent = Pt(18)
    pf.space_before = Pt(4)
    pf.space_after = Pt(8)


def shot(doc: Document, number: int, description: str) -> None:
    """Screenshot placeholder. Left deliberately obvious so nothing is missed."""
    p = doc.add_paragraph()
    run = p.add_run(f"[SCREENSHOT {number}: {description}]")
    run.bold = True
    run.font.color.rgb = RGBColor(0xB0, 0x30, 0x00)
    p.paragraph_format.left_indent = Pt(18)
    p.paragraph_format.space_after = Pt(10)


def table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = h
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, value in enumerate(row):
            cells[i].text = value
    doc.add_paragraph()


def build() -> Document:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    # ---------------------------------------------------------------- title
    title = doc.add_heading("Audio Enhancement for Speech Recordings", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = sub.add_run("AELF - a local-only speech enhancement toolkit")
    run.italic = True
    run.font.size = Pt(13)

    # ------------------------------------------------------ personal details
    heading(doc, "Personal Details", 1)
    table(
        doc,
        ["Field", "Detail"],
        [
            ["Name", NAME],
            ["Email", EMAIL],
            ["Institute", INSTITUTE],
            ["Programme", PROGRAMME],
            ["Enrolment Number", ENROLMENT],
            ["Student ID", STUDENT_ID],
            ["Submission Type", "Solo - individual interview presentation"],
        ],
    )

    # ------------------------------------------------------ executive summary
    heading(doc, "Executive Summary", 1)
    para(
        doc,
        "This report describes AELF, a speech enhancement toolkit built to clean up "
        "noisy recordings without sending any audio off the machine. The project "
        "started from a practical problem: background noise in recorded speech is "
        "easy to describe and awkward to fix, because 'noise' can be a steady hiss, "
        "or several unrelated sounds layered on top of each other, and a method that "
        "handles the first will not handle the second.",
    )
    para(
        doc,
        "The toolkit provides a command-line interface and a browser interface that "
        "share one processing chain, so both produce identical results. It measures "
        "the recording before and after, reports what actually changed, and refuses "
        "to claim an improvement it cannot demonstrate. The browser interface is "
        "pinned to the local machine and Streamlit telemetry is disabled explicitly, "
        "so privacy does not depend on configuration nobody noticed.",
    )
    para(
        doc,
        "The main technical result came from analysing a real 279-second field "
        "recording whose background is not a single steady sound. Its volume stays "
        "constant to within 0.5 dB, but its spectrum shifts by up to 17.9 dB as "
        "different sources pass through. Spectral subtraction can only remove the "
        "component every source shares, so it improves the recording but plateaus. "
        "Measured signal-to-noise ratio rose from 4.09 dB to 9.86 dB and stopped there; "
        "pushing the strength slider higher reduced speech as well as noise.",
    )
    para(
        doc,
        "The honest conclusion is that layered noise needs source separation rather "
        "than filtering. That work is specified and its dependencies are declared, but "
        "it is not implemented, and this report does not claim otherwise. Across the "
        "project, 212 automated tests pass, and three genuine defects found during "
        "verification are documented in Section 5.",
    )

    # ------------------------------------------------------------- tools used
    heading(doc, "Tools Used", 1)

    heading(doc, "Language and Runtime", 2)
    bullet(doc, "Python 3.11.9")
    bullet(doc, "Windows with PowerShell for command-line work")

    heading(doc, "Core Libraries", 2)
    bullet(doc, "NumPy 1.26.4 - array handling for audio frames")
    bullet(doc, "SciPy 1.15.3 - filtering and resampling")
    bullet(doc, "soundfile (libsndfile) 0.12.1 - WAV encoding")
    bullet(doc, "FFmpeg 8.1 - decoding MP3 and M4A input")
    bullet(doc, "Streamlit 1.64.0 - the browser interface")
    bullet(doc, "python-docx - generating this report")

    heading(doc, "Quality Tools", 2)
    bullet(doc, "pytest 9.1.1 - 212 automated tests")
    bullet(doc, "Ruff 0.16.9 - linting, clean on every file")
    bullet(doc, "Git - version control, with meaningful commit messages")

    heading(doc, "Declared but Not Installed", 2)
    para(
        doc,
        "PyTorch, Demucs, SpeechBrain, Silero VAD and faster-whisper are declared in "
        "the project's optional [neural] extra. They are not installed and no code "
        "path depends on them yet. They are listed here as planned scope, not as "
        "tools used.",
    )

    # ------------------------------------------------------------ methodology
    heading(doc, "Methodology", 1)

    heading(doc, "Analysis", 2)
    para(
        doc,
        "The project treats measurement before and after processing as the basis for "
        "every claim, rather than treating the change as an improvement by default. "
        "The analysis layer provides the following.",
    )
    bullet(
        doc,
        "Decoding: input is decoded to a canonical 48 kHz float32 array shaped "
        "[frames, channels], so every later stage works on a single known format "
        "regardless of what the source file was.",
    )
    bullet(
        doc,
        "Levels: peak, RMS, crest factor, dynamic range, DC offset and integrated "
        "LUFS loudness, with clipping counted as discrete events.",
    )
    bullet(
        doc,
        "Noise profiling: a minimum-statistics estimate takes the 10th percentile "
        "across 30 ms frames to approximate the quietest moments, then reports the "
        "floor, the dominant band, spectral tilt and any mains hum.",
    )
    bullet(
        doc,
        "Speech detection: frames are classified as speech or background so noise "
        "can be measured only where there is no speech.",
    )
    bullet(
        doc,
        "Comparison against a clean reference: SDR and SI-SDR, with the important "
        "distinction that SDR is sensitive to overall gain while SI-SDR is not. When "
        "output and reference differ in loudness, only SI-SDR is meaningful, and the "
        "tool says so instead of printing a flattering number.",
    )
    para(
        doc,
        "Processing itself runs in a fixed order: spectral noise suppression, then "
        "optional tidy filters (high-pass, low-pass, speech EQ, de-esser), then "
        "optional loudness normalisation. This chain lives in one shared function "
        "called by both the command line and the browser, which is why the two "
        "interfaces cannot drift apart.",
    )

    heading(doc, "Finding", 2)
    para(
        doc,
        "The decisive experiment was a real recording rather than a synthetic test "
        "tone: war-intercept.wav, 279 seconds, mono, 48 kHz, starting at an "
        "estimated noise floor of -22.68 dBFS and a signal-to-noise ratio of only "
        "2.45 dB as reported by the profiler, with a narrowband component at 66.7 Hz "
        "consistent with mains hum.",
    )
    para(
        doc,
        "The first analysis suggested the background was stationary, which would have "
        "made the existing suppressor a good fit. A deeper frame-by-frame check "
        "contradicted that. Splitting the file into 32 ms frames and comparing each "
        "background frame against the median background spectrum gave the result that "
        "changed the project's direction:",
    )
    table(
        doc,
        ["Measurement of background frames", "Result", "What it means"],
        [
            ["Level variation", "0.50 dB standard deviation", "Volume is steady"],
            ["Spectral shape, mean deviation", "6.68 dB", "Content is not steady"],
            ["Spectral shape, 90th percentile", "13.72 dB", "Often very different"],
            ["Spectral shape, worst frame", "17.89 dB", "Layers, not one hiss"],
        ],
    )
    para(
        doc,
        "Steady volume with changing content is the signature of several sounds "
        "overlapping: the background holds one loudness while different sources "
        "occupy the spectrum in turn. Because the suppressor subtracts a single "
        "median spectrum, it can only remove the portion that every source shares. "
        "That prediction was then tested directly by sweeping the strength parameter "
        "across the real file.",
    )
    table(
        doc,
        ["Strength", "Noise floor (dBFS)", "Floor drop (dB)", "SNR (dB)"],
        [
            ["0.00", "-19.50", "0.00", "4.09"],
            ["0.25", "-20.35", "0.85", "4.51"],
            ["0.50", "-22.72", "3.22", "6.13"],
            ["0.75", "-28.98", "9.48", "9.86"],
            ["1.00", "-30.29", "10.79", "9.62"],
        ],
    )
    para(
        doc,
        "The prediction held. Signal-to-noise ratio improves strongly up to strength "
        "0.75 and then stops: 9.86 dB at 0.75 falls slightly to 9.62 dB at 1.00, even "
        "though the noise floor continues to fall. Beyond that point the setting is "
        "removing speech along with noise. This is why the slider is capped in effect "
        "rather than in name, and why the recommendation for this file is 0.75 to 1.0 "
        "with awareness of the trade-off.",
    )

    heading(doc, "Outcome", 2)
    bullet(
        doc,
        "Delivered: a working toolkit with a command-line interface (enhance, "
        "compare, serve) and a Streamlit browser interface sharing one tested "
        "processing chain.",
    )
    bullet(
        doc,
        "Delivered: measurement-driven reporting, including gain-invariant noise "
        "floor and SNR reporting, silence handling, and a refusal to compare files "
        "of mismatched duration.",
    )
    bullet(
        doc,
        "Delivered: privacy enforced in code rather than assumed - the server binds "
        "only to the loopback interface, telemetry is disabled explicitly, and source "
        "files are protected against overwrite.",
    )
    bullet(
        doc,
        "Result on layered noise: 10.79 dB of floor reduction and SNR from 4.09 dB "
        "to 9.86 dB, with the plateau explained rather than hidden.",
    )
    bullet(
        doc,
        "Not delivered: removal of layered background noise beyond that plateau. "
        "This requires source separation. The dependencies are declared in the "
        "[neural] extra and the approach is specified, but the code is not written, "
        "so no result is claimed for it.",
    )
    bullet(
        doc,
        "Verification: 212 automated tests pass and Ruff reports no issues. Coverage "
        "includes the cases that usually break audio software - digital silence, "
        "single-frame input, mono and stereo, mismatched reference lengths, and "
        "integer overflow in the limiter.",
    )

    # ------------------------------------------------------ step by step
    heading(doc, "Step-by-Step Process", 1)

    heading(doc, "Step 1 - Install the Project", 2)
    para(
        doc,
        "The package is installed in editable mode with the user interface extra, "
        "which keeps source edits live and pulls in Streamlit. The library and its "
        "optional extras are declared in pyproject.toml with a single console entry "
        "point named aelf.",
    )
    code(doc, "python -m venv .venv\n.venv\\Scripts\\Activate.ps1\n"
              "pip install -e \".[ui]\"\npython -c \"import streamlit, aelf; print('ready')\"")
    shot(doc, 1, "Terminal showing the editable install completing and the "
                 "verification line printing 'ready'.")

    heading(doc, "Step 2 - Inspect the Available Commands", 2)
    para(
        doc,
        "The command-line interface exposes three verbs. Each reports its own "
        "options so the intended behaviour is discoverable without reading source.",
    )
    code(doc, "aelf --help\n\n"
              "usage: aelf [-h] {enhance,compare,serve} ...\n\n"
              "Local-only audio enhancement. Nothing leaves your machine.\n\n"
              "positional arguments:\n"
              "  {enhance,compare,serve}\n"
              "    enhance             Clean up a recording\n"
              "    compare             Score output against a clean reference\n"
              "    serve               Open the browser interface")
    shot(doc, 2, "Terminal showing the aelf --help output above.")

    heading(doc, "Step 3 - Run the Analysis", 2)
    para(
        doc,
        "Before processing anything, the recording is measured. On the field "
        "recording used for this report the profiler reports the figures below, "
        "including a 66.7 Hz component standing 104 dB above the noise spectrum, "
        "which is consistent with mains hum.",
    )
    code(doc, "file\n"
              "  duration   279.0s\n"
              "  rate       48000 Hz\n"
              "  channels   1\n"
              "  peak       0.7209\n"
              "  LUFS-I     -15.60   LRA 3.24\n\n"
              "noise profile\n"
              "  noise_floor_dbfs       -22.68\n"
              "  is_stationary          True\n"
              "  estimated_snr_db       2.45\n"
              "  dominant_noise_band_hz (2500.0, 2700.0)\n"
              "  confidence             medium\n"
              "  notes                  ['Narrowband component at 66.7 Hz stands "
              "104 dB above the noise spectrum, consistent with mains hum.']")
    shot(doc, 3, "Terminal showing the analysis output for war-intercept.wav.")

    heading(doc, "Step 4 - Check Whether the Background Is Really Steady", 2)
    para(
        doc,
        "This step is what changed the project. The profiler's stationary flag said "
        "the background was steady, but that flag only compares levels. Splitting the "
        "file into 32 ms frames and measuring how far each background frame deviates "
        "from the median background spectrum shows the level is steady while the "
        "content is not.",
    )
    para(
        doc,
        "The deviation figures in the Finding section come from this check: 6.68 dB "
        "mean, 13.72 dB at the 90th percentile, 17.89 dB for the worst frame. For "
        "comparison, a genuinely steady hiss or hum deviates by only a few dB. This "
        "is the evidence that the background is layered.",
        italic=True,
    )
    shot(doc, 4, "Terminal showing the frame-by-frame stationarity check with the "
                 "spectral deviation figures.")

    heading(doc, "Step 5 - Sweep the Strength Setting", 2)
    para(
        doc,
        "Because the suppressor removes only what all overlapping sources share, "
        "raising the strength setting should improve signal-to-noise ratio up to a "
        "point and then begin damaging speech. Sweeping the parameter on the real "
        "file confirms that prediction exactly.",
    )
    code(doc, "strength  floor   drop   speech    SNR\n"
              "                   dB      dB      dB\n"
              "  0.00   -19.50   0.00   -15.41    4.09\n"
              "  0.25   -20.35   0.85   -15.84    4.51\n"
              "  0.50   -22.72   3.22   -16.59    6.13\n"
              "  0.75   -28.98   9.48   -19.11    9.86\n"
              "  1.00   -30.29  10.79   -20.66    9.62\n\n"
              "SNR peaks at strength 0.75 and then falls, while the floor keeps\n"
              "dropping - the extra suppression is removing speech, not noise.")
    shot(doc, 5, "Terminal showing the strength sweep table.")

    heading(doc, "Step 6 - Enhance a Recording from the Command Line", 2)
    para(doc, "Processing a file, specifying an output and a loudness target:")
    code(doc, "aelf enhance work/uploads/demo_noisy.wav \\\n"
              "    --strength 0.5 --target -23 -o work/outputs/clean.wav")
    para(
        doc,
        "The tool prints a before-and-after report naming each stage it applied, for "
        "example the noise floor, the speech-to-background ratio, loudness and peak "
        "in their respective units. A run on the demo file produced this:",
    )
    code(doc, "What changed\n"
              "- Noise floor: -29.2 dBFS -> -31.6 dBFS (down 2.4 dB (better))\n"
              "- Speech against background: 0.8 dB -> 5.5 dB (up 4.8 dB (better))\n"
              "- Loudness: -22.6 LUFS -> -23.0 LUFS\n"
              "- Peak: -11.9 dBFS -> -6.8 dBFS")
    shot(doc, 6, "Terminal showing the enhance command and its before/after report.")

    heading(doc, "Step 7 - Score the Result Against a Clean Reference", 2)
    para(
        doc,
        "Scoring requires the same speech recorded clean. The comparison refuses to "
        "run when durations do not match, which is why the demo files in this project "
        "are generated as a matched pair rather than trimmed from different takes.",
    )
    code(doc, "aelf compare work/outputs/clean.wav work/uploads/demo_clean_reference.wav")
    para(
        doc,
        "Both SDR and SI-SDR are reported, and the report explains which one to trust: "
        "SDR is sensitive to overall gain, so after loudness normalisation it can look "
        "poor even when the speech is clean. SI-SDR is gain-invariant and is the "
        "meaningful figure.",
    )
    shot(doc, 7, "Terminal showing the compare output with SDR and SI-SDR.")

    heading(doc, "Step 8 - Open the Browser Interface", 2)
    para(
        doc,
        "The browser interface offers the same controls, with a target loudness "
        "slider ranging from -30 to -16 LUFS. For a layered-noise file the practical "
        "settings are strength 0.75 to 1.0 and a target of -30 LUFS.",
    )
    code(doc, "aelf serve\n\n"
              "Starting AELF at http://127.0.0.1:8501\n"
              "The page opens in your browser as soon as it is ready.\n"
              "Press Ctrl+C to stop.")
    para(
        doc,
        "The server binds only to 127.0.0.1 and Streamlit usage statistics are "
        "disabled explicitly in the code rather than relying on a configuration file "
        "being present. The page loads a file, displays the recording's measurements, "
        "plays the original, the processed result and both back to back, and offers "
        "the processed audio as a download.",
    )
    shot(doc, 8, "Browser showing the uploaded file, its measurements, the three "
                 "audio players and the download button.")
    shot(doc, 9, "Browser showing the sidebar with the strength slider at 0.75 and "
                 "target loudness at -30 LUFS.")

    heading(doc, "Step 9 - Verify with Automated Tests", 2)
    para(
        doc,
        "The suite is run before any commit. It covers digital silence, single-frame "
        "input, mono and stereo encoding, mismatched reference durations, limiter "
        "overflow, privacy flags, and a decode of the actual download bytes.",
    )
    code(doc, "python -m pytest\n212 passed in 32.00s\n\n"
              "python -m ruff check app.py src tests scripts\nAll checks passed!")
    shot(doc, 10, "Terminal showing 212 passed and the clean Ruff result.")

    # ------------------------------------------------- defects found
    heading(doc, "Defects Found During Verification", 1)
    para(
        doc,
        "Three faults were found by testing behaviour rather than assuming it. They "
        "are included because verifying the work honestly matters more than the "
        "appearance of a smooth build.",
    )

    heading(doc, "Stereo Playback Failed Because Streamlit Cannot Encode 2-D Float Arrays", 2)
    para(
        doc,
        "Stereo audio passed through st.audio and produced nothing, because Streamlit "
        "encodes float arrays as WAV and a two-dimensional array is not valid WAV. "
        "The page now encodes to 16-bit PCM bytes through a single helper, used for "
        "the players and the download, so the audio played and the audio downloaded "
        "come from the same bytes.",
    )

    heading(doc, "The Download Test Asserted Nothing About the Download", 2)
    para(
        doc,
        "The original test confirmed a download button was present, which would have "
        "passed even if the button handed over bytes no player could open. Streamlit's "
        "test harness reports whether a button was pressed but never its payload, so "
        "the test was replaced with three that decode the actual bytes: a ramp "
        "round-trips at the right rate and length, a signal that would overrun comes "
        "back under the -1 dBFS ceiling, and stereo keeps two distinct channels.",
    )

    heading(doc, "The Browser Opened Before the Server Was Listening", 2)
    para(
        doc,
        "The first run reported ERR_CONNECTION_REFUSED, which looked like a broken "
        "application. The cause was a fixed 1.5-second timer that opened the browser "
        "unconditionally, while Streamlit binds in about two seconds on a warm start "
        "and can take fifteen on a cold one. The fix polls the port until something "
        "answers, then opens the browser once. If the server never starts, no browser "
        "window opens at all. Two tests reproduce the original failure.",
    )

    # ------------------------------------------------- limitations
    heading(doc, "Limitations and Future Work", 1)
    para(
        doc,
        "The layered background in the test recording is not fully removed, and the "
        "measurement above shows why rather than leaving it open. Remaining work, in "
        "priority order:",
    )
    numbered(
        doc,
        "Source separation for layered noise. A separation model isolates voices and "
        "background sources directly instead of subtracting one averaged spectrum, "
        "which is the correct approach for the layered case measured in this report.",
    )
    numbered(
        doc,
        "Voice activity detection, so noise is measured and removed only where "
        "someone is not speaking, improving both the estimate and the artefact "
        "control.",
    )
    numbered(
        doc,
        "Transcription as an intelligibility check, giving a functional measure that "
        "does not depend on a clean reference recording.",
    )
    para(
        doc,
        "Validation is also narrower than it should be. Correctness was confirmed on "
        "synthetic test tones and on one real recording; no clean reference exists for "
        "the field recording, so its figures are relative measurements rather than "
        "scored results.",
    )

    # ------------------------------------------------- screenshot list
    doc.add_page_break()
    heading(doc, "Screenshot Checklist", 1)
    para(
        doc,
        "Ten placeholders appear above. Capture them in order and place each next to "
        "its marker.",
        italic=True,
    )
    for n, desc in [
        (1, "Terminal - editable install and the 'ready' confirmation"),
        (2, "Terminal - aelf --help"),
        (3, "Terminal - analysis output for war-intercept.wav"),
        (4, "Terminal - frame-by-frame stationarity check, spectral deviations"),
        (5, "Terminal - strength sweep table"),
        (6, "Terminal - aelf enhance with before/after report"),
        (7, "Terminal - aelf compare with SDR and SI-SDR"),
        (8, "Browser - uploaded file, measurements, three players, download button"),
        (9, "Browser - sidebar with strength 0.75 and target -30 LUFS"),
        (10, "Terminal - 212 passed and clean Ruff result"),
    ]:
        bullet(doc, f"Screenshot {n}: {desc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return doc


if __name__ == "__main__":
    document = build()
    paras = len(document.paragraphs)
    tables = len(document.tables)
    print(f"wrote {OUT}")
    print(f"{OUT.stat().st_size / 1024:.1f} KB, {paras} paragraphs, {tables} tables")
