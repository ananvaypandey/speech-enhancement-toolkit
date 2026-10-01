"""Typed error hierarchy.

Every error carries a `user_message` that is safe to render verbatim in the
Streamlit UI, plus a `remedy` describing what the operator should do. The
distinction matters: a raw traceback tells the user nothing actionable, and
silently swallowing an error would misrepresent the state of the processing.
"""

from __future__ import annotations


class AelfError(Exception):
    """Base class for all toolkit errors."""

    user_message: str = "An unexpected processing error occurred."
    remedy: str = "Check the logs for details and retry."

    def __init__(self, detail: str = "", *, user_message: str | None = None, remedy: str | None = None) -> None:
        self.detail = detail
        if user_message is not None:
            self.user_message = user_message
        if remedy is not None:
            self.remedy = remedy
        super().__init__(detail or self.user_message)

    def render(self) -> str:
        return f"{self.user_message}\n\n{self.remedy}"


class UnsupportedFormatError(AelfError):
    user_message = "This file format is not supported."
    remedy = "Convert the file to WAV, MP3, or M4A and upload it again."


class DecodeError(AelfError):
    user_message = "The audio file could not be decoded."
    remedy = "The file may be corrupt or truncated. Verify it plays in a standard player, then retry."


class EmptyAudioError(AelfError):
    user_message = "The audio file contains no samples."
    remedy = "The file is zero-length. Provide a recording that contains actual audio."


class SourceIntegrityError(AelfError):
    user_message = "Refusing to write: the output path would overwrite the source audio."
    remedy = "This is a safety guard. Choose an output directory separate from the input file."


class SourceMutatedError(AelfError):
    user_message = "The source file changed during processing and the run was aborted."
    remedy = "Do not edit the input file while it is being processed. Reload and retry."


class MisalignedReferenceError(AelfError):
    user_message = "The reference audio does not match the processed audio in length, so quality metrics were not computed."
    remedy = (
        "Reference-based metrics (SI-SDR, STOI, PESQ) require identical or near-identical "
        "timing. Trim or pad the reference so its duration matches the input, or omit the "
        "reference to receive reference-free metrics only."
    )


class ReferenceUnavailableError(AelfError):
    user_message = "No reference audio was supplied, so perceptual quality metrics cannot be computed."
    remedy = (
        "Metrics such as PESQ, STOI and SI-SDR need a clean reference recording. Without one, "
        "only reference-free measurements (SNR, spectral distance, noise floor) are reported."
    )


class MetricUnavailableError(AelfError):
    """A quality metric could not be computed for a legitimate reason.

    Distinct from `BackendUnavailableError`: nothing is missing or broken, the
    inputs simply do not support the measurement. Reported rather than guessed.
    """

    user_message = "No quality score was computed, because the inputs do not support one."
    remedy = (
        "Reference-based metrics need a clean recording of the same speech. Without a usable "
        "reference, only measurements of the audio itself are reported."
    )


class BackendUnavailableError(AelfError):
    user_message = "The requested processing backend is not available."
    remedy = "Install the optional dependency, or select a different backend in the interface."


class EnhancementError(AelfError):
    user_message = "Speech enhancement failed."
    remedy = "Retry with a lower processing strength, or switch to the spectral fallback backend."


class SeparationError(AelfError):
    user_message = "Source separation failed."
    remedy = "Retry, or select the alternate separation backend."


class TranscriptionError(AelfError):
    user_message = "Transcription failed."
    remedy = "Try a smaller model size, or verify the audio contains intelligible speech."


class VadError(AelfError):
    user_message = "Voice activity detection failed."
    remedy = "The energy-based fallback was also unavailable. Verify the audio is decodable."
