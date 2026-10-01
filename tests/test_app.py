"""Headless smoke test of the Streamlit app.

Streamlit runs a page's script top to bottom on every interaction, so the
cheapest way to check it is to drive that script directly with Streamlit's own
testing harness rather than clicking through a browser. A pass here means the
page renders without raising, which is the failure mode this exists to catch: a
bad import, a missing config constant, a None where a number was expected.

Note on what is assertable: the harness exposes widgets (`button`,
`download_button`, `file_uploader`, `slider`, ...) but has no accessor for
`st.audio`, so the presence of the players is verified by looking for the
elements around them rather than pretending to inspect them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from aelf.config import PATHS  # noqa: E402
from conftest import SR, speech_like, white_noise  # noqa: E402

pytest.importorskip("streamlit", reason="Streamlit is an optional UI dependency")
AppTest = pytest.importorskip("streamlit.testing.v1", reason="Streamlit testing harness").AppTest

APP = ROOT / "app.py"


@pytest.fixture(scope="module")
def noisy_wav(tmp_path_factory) -> Path:
    speech = speech_like(3.0, SR, 120.0, 4) * 0.3
    hiss = white_noise(3.0, SR, 0.012, seed=3)
    path = tmp_path_factory.mktemp("upload") / "sample_noisy.wav"
    sf.write(str(path), (speech + hiss).astype(np.float32)[:, None], SR, subtype="PCM_16")
    return path


def _has_text(elements, needle: str) -> bool:
    """Whether any rendered element carries `needle`.

    Streamlit messages are single strings, so this is just a substring check
    across the set; kept as a helper so the assertions read as intent.
    """
    return any(needle in str(getattr(el, "value", "")) for el in elements)


def _body_elements(at) -> list[str]:
    return [getattr(el, "value", "") or str(getattr(el, "label", "")) for el in at.markdown] + [
        getattr(el, "value", "") or str(getattr(el, "caption", "")) for el in at.caption
    ]


def run(upload_path: Path | None = None, strength: float = 0.5, normalise: bool = False) -> AppTest:
    """Drive the page: run once to draw it, set controls, upload, run again.

    The first run is required because widget handles only exist after the script
    has executed far enough to create them, and `st.stop()` on the no-upload path
    means the uploader is absent until that path is passed.
    """
    PATHS.ensure()
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    if not len(at.sidebar.slider):
        return at
    at.sidebar.slider[0].set_value(strength)
    at.sidebar.checkbox[0].set_value(normalise)
    if upload_path is not None:
        at.file_uploader[0].upload(upload_path.name, upload_path.read_bytes())
    at.run()
    return at


class TestLandingPage:
    def test_renders_before_any_upload(self) -> None:
        """The first thing anyone sees. It must not crash, and it must say what
        to do next rather than sitting empty."""
        at = run()
        assert not list(at.exception), list(at.exception)
        assert len(at.title) == 1
        assert len(at.info) == 1, "expected a prompt to add a file"
        assert _has_text(at.info, "file"), "the prompt should say to add a file"

    def test_states_that_audio_stays_local(self) -> None:
        """The privacy claim is the reason to use this over a web service, so
        it belongs on the page rather than in a README nobody opens."""
        at = run()
        assert not list(at.exception), list(at.exception)
        captions = " ".join(str(c.value) for c in at.caption)
        assert "not uploaded" in captions or "locally" in captions


class TestProcessing:
    def test_processes_an_uploaded_file(self, noisy_wav: Path) -> None:
        """File in, cleaned audio out. The whole point of the page."""
        at = run(noisy_wav)
        assert not list(at.exception), list(at.exception)
        assert len(at.get("audio")) >= 2, "expected a player for the original and one for the result"

    def test_reports_what_changed(self, noisy_wav: Path) -> None:
        """The before/after numbers are the page's substance. They must appear,
        and they must be measured from the audio rather than hard-coded."""
        at = run(noisy_wav, strength=0.75)
        assert not list(at.exception), list(at.exception)
        body = " ".join(_body_elements(at))
        assert "Noise floor" in body
        assert "What changed" in body
        # A real dB figure, not a placeholder.
        assert "dBFS" in body

    def test_offers_a_download_of_the_cleaned_audio(self, noisy_wav: Path) -> None:
        at = run(noisy_wav)
        assert not list(at.exception), list(at.exception)
        assert len(at.get("download_button")) >= 1, "expected a download control"

    def test_strength_zero_is_not_an_error(self, noisy_wav: Path) -> None:
        """Strength 0 means "leave it alone". That is a valid request, and the
        page must treat it as one rather than as a missing setting."""
        at = run(noisy_wav, strength=0.0)
        assert not list(at.exception), list(at.exception)
        assert not list(at.error), [e.value for e in at.error]

    def test_maximum_strength_renders(self, noisy_wav: Path) -> None:
        at = run(noisy_wav, strength=1.0)
        assert not list(at.exception), list(at.exception)
        assert not list(at.error), [e.value for e in at.error]

    def test_normalisation_toggle_runs(self, noisy_wav: Path) -> None:
        at = run(noisy_wav, normalise=True)
        assert not list(at.exception), list(at.exception)
        assert not list(at.error), [e.value for e in at.error]

    def test_warns_when_the_source_is_already_clipped(self) -> None:
        """Clipping in the source is unrecoverable. Saying so up front stops the
        user blaming the cleanup for a problem it cannot fix."""
        clipped = np.full(int(SR * 1.0), 0.999, dtype=np.float32)
        clipped[::100] = -1.0
        path = Path(PATHS.uploads) / "clipped.wav"
        PATHS.ensure()
        sf.write(str(path), clipped[:, None], SR, subtype="PCM_16")

        at = run(path)
        assert not list(at.exception), list(at.exception)
        assert len(at.warning) >= 1, "expected a warning about unrecoverable clipping"
        assert _has_text(at.warning, "undone") or _has_text(at.warning, "full scale")

    def test_explains_a_bad_file_instead_of_crashing(self, tmp_path: Path) -> None:
        """A corrupt upload must produce an explanation, not a traceback."""
        bad = tmp_path / "broken.wav"
        bad.write_bytes(b"this is not audio" * 200)
        at = run(bad)
        assert not list(at.exception), list(at.exception)
        assert len(at.error) >= 1, "expected an error message for an unreadable file"


class TestClaims:
    def test_does_not_claim_a_quality_score(self, noisy_wav: Path) -> None:
        """The project's core rule: no reference, no score. The page must not
        imply otherwise by printing one."""
        at = run(noisy_wav)
        assert not list(at.exception), list(at.exception)
        body = " ".join(_body_elements(at)).lower()
        for claimed in ("sdri", "sdr:", "quality score of", "improved by", "98%", "excellent"):
            assert claimed not in body, f"page claims {claimed!r} without a reference"

    def test_points_at_the_reference_command_when_a_score_is_wanted(
        self, noisy_wav: Path
    ) -> None:
        """Telling the user how to get a real score is more useful than a
        disclaimer alone."""
        at = run(noisy_wav)
        assert not list(at.exception), list(at.exception)
        body = " ".join(_body_elements(at))
        assert "compare" in body or "reference" in body.lower()
