# Challenge 3: Denoising and Transcribing Intercepted Audio

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

It also scores how much improvement actually happened, when you have a clean
recording to compare against.

Separating two people talking, detecting when speech starts and stops, and
transcribing it are planned but not built yet.

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
├── cli.py               Command line: enhance, compare, serve
├── io/                  Getting audio in, putting audio out
│   ├── decode.py        WAV/MP3/M4A → one standard format
│   ├── encode.py        Write results, with a no-distortion guarantee
│   └── integrity.py     Fingerprints proving your originals are untouched
├── analysis/            Looking at audio without changing it
│   ├── levels.py        Loudness, peaks, distortion counting
│   ├── noise_profile.py What's underneath the voice
│   └── probe.py         What the file claims about itself
├── enhance/             Removing background noise
│   └── spectral.py      OMLSA masking, pure numpy — nothing to download
├── postprocess/         Changing audio, after enhancement
│   ├── filters.py       Rumble removal, speech EQ, harsh-sound control
│   └── normalize.py     Even volume, guaranteed no clipping
└── evaluate/            Scoring output against a known-good reference
    └── compare.py       SDR, SI-SDR, and an explanation when neither applies

app.py                   The browser interface
```

Still to come: `separate/`, `transcribe/`, `viz/`.

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

### Removing the background noise

The noise remover estimates the background hiss from the quietest moments in
your file, works out which frequencies hold speech and which hold noise, and
turns that down. No neural network, nothing to download, and it runs in about a
hundredth of the recording's duration.

It is good at steady background noise — hiss, hum, fan whir, air conditioning.
It is not good at noise that moves: a passing car, someone talking in the next
room, a keyboard. A trained model handles those; this method cannot, and the
toolkit says so rather than pretending otherwise.

There is a `strength` control for a reason. Turned up, it removes more noise and
also takes a little brightness out of the voice. On a clean recording it turns
itself off entirely and hands the audio straight back — the tool checks whether
there is anything to remove before touching anything.

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

pip install -e .
pip install -e ".[ui]"
```

FFmpeg (needed for MP3 and M4A; WAV works without it):

```powershell
winget install Gyan.FFmpeg
```

---

## Using it

### In the browser

The easiest way in. This starts a local page and opens it:

```powershell
python -m aelf.cli serve
```

Then: add your file, pick a strength, listen, download. The audio never leaves
your computer.

### From the command line

```powershell
# Clean up a file. Result lands in work\outputs\
python -m aelf.cli enhance my_recording.wav

# Choose where it goes, and how hard it works
python -m aelf.cli enhance my_recording.wav -o clean.wav --strength 0.8

# Also even out the volume while you're at it (-16 suits headphones)
python -m aelf.cli enhance my_recording.mp3 --target -16
```

`--strength` runs from `0.0` (leave it alone) to `1.0` (remove as much as this
method can without hollowing out speech). `0.5` is the default and a sensible
starting point.

### Measuring real improvement

If you happen to have the clean recording of the same speech, made the same way,
you can get an actual score instead of a measurement:

```powershell
python -m aelf.cli compare my_recording_clean.wav my_recording_clean_reference.wav
```

It reports signal-to-distortion ratio and its scale-invariant variant, and it
refuses to produce a number if the two files don't line up — see
[Honest by design](#honest-by-design).

---

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q         # 174 tests
.\.venv\Scripts\python.exe -m ruff check app.py src tests scripts
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

### How the noise remover was tuned

Every number in this project was measured, not assumed, and the noise remover
took three rounds of it. The bugs are worth recording, because each one looked
correct in code and was only exposed by measurement.

**Reconstruction blew up by a factor of a million.** A Hann window is exactly
zero at its first and last sample, so the very first output sample was being
divided by a window weight of about `1.8e-11`. Peaks came back at `7e+06`
instead of `0.10`. Fixed by padding half a frame at both ends and trimming it
back off afterwards. There is now a test that reconstructs a signal through the
overlap-add path with an all-ones mask and requires it to come back within
`1e-6` of the original — that test fails loudly if the windowing ever breaks
again.

**The mask removed almost nothing.** The noise estimator took the 15th percentile
of the quietest frames, which on real hiss measured **14.7 dB below the true
noise power**. An underestimated noise floor inflates the apparent
signal-to-noise ratio, which pushes the decision-directed estimator's fixed point
toward unity gain — so asking for more suppression did essentially nothing.
Measured: 0.8 dB of hiss removed at strength 0.5. Switching to the *mean* of
those frames: 12 dB.

**Then it damaged clean recordings.** With suppression actually working, a clean
voice recording came back changed by 8.4% of its own RMS, which is audible. The
cause is inherent to the approach: on clean audio the "quietest frames" are the
speaker's quietest moments, so the estimator learns speech as noise and then
suppresses it.

The fix is to notice when there is nothing to remove. How far the noise estimate
sits below the overall signal level separates the cases cleanly — 0.8 dB for
noise alone, 7.2 dB for speech at 22 dB SNR, **28.1 dB for clean speech**. Past
about 26 dB the honest conclusion is that the quietest thing in the file is the
speaker, not the room, so suppression switches itself off. Clean recordings now
come back bit-identical, while speech at 22 dB SNR still gains 18.8 dB.

The browser interface is tested by driving the real Streamlit script headlessly,
which catches the failures that matter there — a bad import, a config value that
moved, a stereo buffer the audio player can't encode.

---

## Planned

- Source separation (Demucs `htdemucs`, SpeechBrain SepFormer as fallback)
- Voice activity detection
- Transcription with timestamps, on original and enhanced audio for comparison
- A neural noise-removal backend, used automatically when installed
- Waveform and spectrogram views

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