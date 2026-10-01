# AELF — Speech Enhancement Toolkit

Turns a noisy recording into a cleaner one, entirely on your own computer.

Nothing you feed it ever leaves your machine. There is no upload step, no
telemetry, and no account. The only thing that ever touches the internet is the
one-time download of the model files, which happens on first run and is
optional if you only want the classic signal-processing path.

---

## What it does

Point it at a WAV, MP3, or M4A file and it will:

1. **Look at the file first** — what format it really is, how long, how many
   channels, whether the header lies about any of that.
2. **Measure it** — how loud, how noisy, whether any part is distorting, how
   much headroom is left before clipping.
3. **Clean it** — remove the constant hiss and rumble underneath the voice,
   then even out the volume.
4. **Tidy it** — cut the low rumble that adds no speech, lift the frequencies
   the ear uses to understand speech, take the edge off harsh "sss" sounds.
5. **Report honestly** — what it changed, what it measured, and — this is the
   important part — what it *couldn't* verify.

If you want, it will also separate two people talking, detect when speech
starts and stops, transcribe it to text, and score how much improvement
actually happened.

---

## Honest by design

This is the part that sets the project apart, so it's worth reading twice.

Most tools of this kind will show you a "noise reduction: 87%" and call it a
day. That number is usually measuring the wrong thing. We refuse to.

**No metric is reported unless it can be trusted.** Concretely:

| Situation | What we do |
|---|---|
| No clean reference track to compare against | Report only measurements that need no reference (loudness, noise floor, distortion count). Say plainly that improvement *could not be verified*. |
| Reference track is the wrong length | Refuse to compare. Silently stretching audio to make it fit would produce a confident, meaningless number. |
| Reference is off by more than 50 ms | Refuse. Past that offset any comparison is measuring the misalignment, not the audio. |
| Two speakers too close in level to tell apart | Report low confidence and explain why, instead of claiming a clean split. |
| A "separated" stem is nearly silent | Treat as degenerate. A stem that quiet didn't get separated; it got discarded. |
| No transcript to compare against | Report no word-error rate. There is nothing to score against. |

Reference-based measurements (signal-to-distortion, intelligibility scores) are
always labelled separately from reference-free ones, and are only ever shown
when a valid, aligned reference exists.

---

## How it's built

### One job per module

Each piece does one thing and is tested on its own before anything else exists.
If the noise remover breaks, you know it's the noise remover, not the user
interface.

```
src/aelf/
├── config.py            Central settings and pinned constants
├── types.py             Shared data shapes
├── errors.py            Errors that carry a message worth reading
├── io/                  Getting audio in, putting audio out
│   ├── decode.py        WAV/MP3/M4A → one standard format
│   ├── encode.py        Write results, with a no-distortion guarantee
│   └── integrity.py     Fingerprints proving your originals are untouched
├── analysis/            Looking at audio without changing it
│   ├── levels.py        Loudness, peaks, distortion counting
│   ├── noise_profile.py What's underneath the voice
│   └── probe.py         What the file claims about itself
└── postprocess/         Changing audio, after enhancement
    ├── filters.py       Rumble removal, speech EQ, harsh-sound control
    └── normalize.py     Even volume, guaranteed no clipping
```

Still to come: `enhance/`, `separate/`, `transcribe/`, `evaluate/`, `viz/`,
and the Streamlit interface.

### Why everything is forced to one format

Loaded audio is converted to **48 kHz, 32-bit float** immediately. Downstream
code then only ever handles one sample rate and one numeric type.

This isn't tidiness for its own sake. Sample rate conversion is lossy, and
doing it once, up front, means every later measurement compares like with like.
It also means the loudness math below can be checked against the official
standard rather than against an approximation.

### Your originals are protected, and provably so

Three independent guards:

1. Uploads and outputs live in **separate directories**. Nothing is ever
   written where an input was.
2. Before writing, the toolkit recomputes a **SHA-256 fingerprint** of the
   source and compares it to the fingerprint taken at load. If the file
   changed underneath us, the write is refused.
3. The output path is checked against the source path. Collision is an error,
   not an overwrite.

The fingerprints are also useful to you: they prove a file hasn't changed since
it was processed.

### Loudness follows the official standard

Loudness is measured using **EBU R128 / ITU-R BS.1770** — the same standard
used by broadcast and streaming platforms. This means a level of `-23 LUFS`
here means the same thing it means in a professional audio tool.

Getting this right took real work. The filter curve that models human hearing
has to be correct, and an early version was subtly wrong. It's now verified
against `pyloudnorm`, an independent implementation, and matches exactly:

```powershell
python scripts/check_lufs_reference.py
```

The loudness filter is also applied at the correct frequency rather than
approximated, which is why the implementation measures in two stages instead
of one.

### Never louder than it should be

Processing adds gain — boosting speech frequencies can push a recording past
full scale, which sounds like crackling. Two independent safeguards:

- **The gain-limiting stage** scales back rather than distorting.
- **The writer** checks again, and if anything still exceeds the ceiling, holds
  it there. Output is capped at **-1 dBFS**, leaving a little room so later
  conversion to MP3 can't push it over either.

`-1 dBFS` is not arbitrary: it is `0.891251` in linear terms. An earlier
version had this constant wrong (it allowed about -0.1 dBFS), which let
processed files distort. There are now tests that hold the ceiling across
several kinds of loud input, and a script you can run yourself:

```powershell
python scripts/check_clipping.py
```

### The harsh-sound control (de-esser)

"Ssss" and "shh" sounds in English sit in the 5.5–9 kHz region. Leave them
alone and they read as harshness rather than as the speaker, because they
survive noise removal while the noise around them does not.

This stage softens that band only, and only when there's real sibilance to
tame.

Two things it deliberately does **not** do:

- It does not use a fixed loudness threshold. A fixed threshold can't work:
  the frequency band you care about sits far quieter than the full recording,
  so an absolute trigger never fires, and the stage quietly does nothing. (This
  was a real bug. The output was byte-for-byte identical to the input.)
- It keys to the band's own average level instead, which means it behaves the
  same whether someone whispered or shouted. Verified: identical reduction
  across 100 dB of input volume.

It also checks there's genuine sibilance present before touching anything, so
low-frequency material passes through untouched.

---

## Getting started

Requires **Python 3.11** and **FFmpeg**. Python 3.11 specifically, because
some of the audio libraries we use don't provide Windows wheels for anything
newer.

```powershell
git clone <your-repo-url>
cd speech-enhancement-toolkit

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

FFmpeg (needed for MP3 and M4A; WAV works without it):

```powershell
winget install Gyan.FFmpeg
```

---

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q         # 124 tests
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe scripts/check_clipping.py
.\.venv\Scripts\python.exe scripts/check_lufs_reference.py
```

### On testing

Tests aren't paperwork here, they're how we found the bugs that mattered. Two
examples worth knowing about, because they shaped how the suite is written:

**A test that couldn't fail.** The de-esser originally had a test using steady
hissing noise as its input. That was wrong — hissing noise isn't a harsh sound,
and a correctly-behaving harshness control is *supposed* to leave it alone. The
test would have passed against a completely broken stage. The fixture is now
band-limited noise with a sharp attack and decay, which is what an actual "sss"
sounds like. This is recorded in the test's docstring so it doesn't get
simplified back later.

**Measuring the wrong quantity.** The speech EQ test compared energy between
two frequency bands. That can't detect a gain error, because a filter that
doubled its gain changes *both* bands. The test now reads the filter's actual
frequency response, which caught a genuine bug: filters were applying twice the
requested gain, because running a filter forwards and backwards (to keep
timing intact) applies it twice.

Where a measurement can be done more directly than by comparison, we do it
directly. Filter gains are verified from the filter's response to a single
impulse, not inferred.

---

## Planned

- Noise suppression (spectral, with a neural option)
- Source separation (Demucs `htdemucs`, SpeechBrain SepFormer as fallback)
- Voice activity detection
- Transcription with timestamps, on original and enhanced audio for comparison
- Objective scoring, gated on having a valid reference
- Waveform and spectrogram views
- Streamlit interface

Where a stage can't verify its own output, it reports that it couldn't rather
than returning a number that looks like an answer.

---

## Requirements

- Python 3.11
- FFmpeg 8.1 or newer
- About 6 GB of free disk for models
- 8 GB RAM minimum; a CUDA GPU helps but isn't required

Deep learning stages run on CPU by default. A 6 GB GPU is enough to speed up
transcription, but the toolkit is built to work without one.

## Licence

Not yet chosen. Add one before publishing publicly.