import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from groq import Groq
import librosa
import numpy as np

FILLER_WORDS = [
    "um",
    "uh",
    "er",
    "ah",
    "like",
    "basically",
    "actually",
    "literally",
]


def _extract_audio_to_wav(media_path: Path, output_wav: Path) -> None:
    """Extract mono 16kHz WAV from a video or audio file using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(media_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0 or not output_wav.exists():
        raise RuntimeError(f"ffmpeg audio extraction failed: {proc.stderr[-400:]}")


def _count_filler_words(transcript: str) -> tuple[int, dict[str, int]]:
    """Count filler words in transcript using word boundaries."""
    lowered = transcript.lower()
    counts: dict[str, int] = {}
    total = 0
    for word in FILLER_WORDS:
        matches = re.findall(rf"\b{re.escape(word)}\b", lowered)
        if matches:
            counts[word] = len(matches)
            total += len(matches)
    return total, counts


def _analyze_waveform(wav_path: Path) -> dict[str, Any]:
    """Compute duration, pitch variation, and audio clarity score from WAV."""
    y, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    duration_sec = float(librosa.get_duration(y=y, sr=sr))

    if len(y) == 0 or duration_sec <= 0.1:
        return {
            "duration_sec": round(duration_sec, 1),
            "pitch_std_hz": 0.0,
            "tone_score": 0.0,
            "audio_clarity_score": 0.0,
        }

    # Pitch (f0) estimation via YIN
    f0 = librosa.yin(y, fmin=65.0, fmax=400.0, sr=sr)
    voiced_f0 = f0[(f0 > 70.0) & (f0 < 380.0)]
    pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 3 else 0.0

    # Expressive speech typically has ~18-65 Hz std; too flat (<10 Hz) sounds monotone
    if pitch_std < 8.0:
        tone_score = max(25.0, (pitch_std / 8.0) * 60.0)
    elif pitch_std <= 75.0:
        tone_score = min(100.0, 65.0 + ((pitch_std - 8.0) / 67.0) * 35.0)
    else:
        tone_score = max(60.0, 100.0 - ((pitch_std - 75.0) / 80.0) * 30.0)

    # RMS energy & dynamic range as a proxy for clear speech vs. low/noisy signal
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    p10_rms = float(np.percentile(rms, 10))
    p90_rms = float(np.percentile(rms, 90))
    dynamic_ratio = p90_rms / max(p10_rms, 1e-6)

    level_score = min(60.0, (mean_rms / 0.05) * 60.0) if mean_rms > 1e-4 else 10.0
    snr_proxy_score = min(40.0, (min(dynamic_ratio, 20.0) / 20.0) * 40.0)
    clarity_score = round(max(0.0, min(100.0, level_score + snr_proxy_score)), 1)

    return {
        "duration_sec": round(duration_sec, 1),
        "pitch_std_hz": round(pitch_std, 1),
        "tone_score": round(float(tone_score), 1),
        "audio_clarity_score": clarity_score,
    }


def _compute_pace_score(wpm: float) -> float:
    """Score speaking rate (0-100); ideal interview pace is ~115-160 WPM."""
    if wpm <= 0:
        return 0.0
    if 115.0 <= wpm <= 160.0:
        return 100.0
    if wpm < 115.0:
        return round(max(20.0, 100.0 - ((115.0 - wpm) / 115.0) * 80.0), 1)
    return round(max(20.0, 100.0 - ((wpm - 160.0) / 90.0) * 80.0), 1)


def analyze_video_audio(media_path: str) -> dict[str, Any]:
    """Extract audio, transcribe via Groq Whisper, and compute speech & audio metrics."""
    path = Path(media_path)
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in environment or .env")

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "extracted_audio.wav"
        _extract_audio_to_wav(path, wav_path)

        wave_stats = _analyze_waveform(wav_path)

        client = Groq(api_key=api_key)
        with open(wav_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(wav_path.name, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )

        transcript_text = str(transcription).strip()
        words = re.findall(r"\b\w+(?:'\w+)?\b", transcript_text)
        word_count = len(words)
        duration_min = max(wave_stats["duration_sec"] / 60.0, 1e-3)
        wpm = round(word_count / duration_min, 1) if word_count > 0 else 0.0

        filler_count, filler_breakdown = _count_filler_words(transcript_text)
        pace_score = _compute_pace_score(wpm)
        filler_penalty = min(40.0, filler_count * 4.0)
        filler_score = round(max(0.0, 100.0 - filler_penalty), 1)

        # Combine pace (40%), tone expressiveness (30%), and low filler usage (30%)
        speech_delivery_score = round(
            0.40 * pace_score + 0.30 * wave_stats["tone_score"] + 0.30 * filler_score,
            1,
        )

        return {
            "transcript": transcript_text,
            "word_count": word_count,
            "duration_sec": wave_stats["duration_sec"],
            "wpm": wpm,
            "pace_score": pace_score,
            "filler_count": filler_count,
            "filler_breakdown": filler_breakdown,
            "pitch_std_hz": wave_stats["pitch_std_hz"],
            "tone_score": wave_stats["tone_score"],
            "audio_clarity_score": wave_stats["audio_clarity_score"],
            "speech_delivery_score": speech_delivery_score,
        }


def analyze_audio_file(audio_path: str) -> dict[str, Any]:
    """Analyze a standalone audio file (such as a live browser mic recording)."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in environment or .env")

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / "normalized_audio.wav"
        _extract_audio_to_wav(path, wav_path)

        wave_stats = _analyze_waveform(wav_path)

        client = Groq(api_key=api_key)
        with open(wav_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(wav_path.name, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="text",
            )

        transcript_text = str(transcription).strip()
        words = re.findall(r"\b\w+(?:'\w+)?\b", transcript_text)
        word_count = len(words)
        duration_min = max(wave_stats["duration_sec"] / 60.0, 1e-3)
        wpm = round(word_count / duration_min, 1) if word_count > 0 else 0.0

        filler_count, filler_breakdown = _count_filler_words(transcript_text)
        pace_score = _compute_pace_score(wpm)
        filler_penalty = min(40.0, filler_count * 4.0)
        filler_score = round(max(0.0, 100.0 - filler_penalty), 1)

        speech_delivery_score = round(
            0.40 * pace_score + 0.30 * wave_stats["tone_score"] + 0.30 * filler_score,
            1,
        )

        return {
            "transcript": transcript_text,
            "word_count": word_count,
            "duration_sec": wave_stats["duration_sec"],
            "wpm": wpm,
            "pace_score": pace_score,
            "filler_count": filler_count,
            "filler_breakdown": filler_breakdown,
            "pitch_std_hz": wave_stats["pitch_std_hz"],
            "tone_score": wave_stats["tone_score"],
            "audio_clarity_score": wave_stats["audio_clarity_score"],
            "speech_delivery_score": speech_delivery_score,
        }


if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv

    load_dotenv()
    if len(sys.argv) < 2:
        print("Usage: python modules/audio.py <path_to_video_or_audio>")
        sys.exit(1)

    result = analyze_video_audio(sys.argv[1])
    for k, v in result.items():
        if k != "transcript":
            print(f"{k}: {v}")
    print("\nTranscript:\n", result["transcript"])