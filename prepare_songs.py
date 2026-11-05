#!/usr/bin/env python3
"""Verify licensed sources and create the frozen 8-second study excerpts."""
import hashlib
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 22050
SECONDS = 8
TARGET_PEAK = 10 ** (-1 / 20)  # -1 dBFS
TRACKS = [
    {
        "source": "songs/calm_blue_lake.mp3",
        "output": "songs/study/calm_blue_lake.wav",
        "start": 48,
        "sha256": "6ed4c5563d90e3a766b7ead1fb630bf862f379e9cd43b8f11490e63714fa5b4c",
    },
    {
        "source": "songs/acoustic_shuffle.mp3",
        "output": "songs/study/acoustic_shuffle.wav",
        "start": 50,
        "sha256": "23c61c8df9d664f4b2cca2681367ad2f232afd716451e501eaf7d612dd49188f",
    },
    {
        "source": "songs/heavy_drums_bass.mp3",
        "output": "songs/study/heavy_drums_bass.wav",
        "start": 38,
        "sha256": "4b6c610723b436a367ec8d7edd23ba8852d70ece59b2e16d8e175dcb648e9c71",
    },
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for item in TRACKS:
        source, output = ROOT / item["source"], ROOT / item["output"]
        actual = digest(source)
        if actual != item["sha256"]:
            raise ValueError(f"checksum mismatch for {source}: {actual}")
        audio, _ = librosa.load(source, sr=SAMPLE_RATE, mono=True,
                                offset=item["start"], duration=SECONDS)
        expected = SAMPLE_RATE * SECONDS
        if len(audio) != expected:
            raise ValueError(f"short excerpt for {source}: {len(audio)} samples")
        audio = np.asarray(audio, dtype=np.float32)
        audio *= TARGET_PEAK / (np.max(np.abs(audio)) + 1e-9)
        output.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output, audio, SAMPLE_RATE, subtype="PCM_16")
        print(f"wrote {output.relative_to(ROOT)} from {item['start']}s")

    # Held-out deterministic signal for adapter calibration; never a study song.
    seconds = 4
    t = np.arange(SAMPLE_RATE * seconds, dtype=np.float32) / SAMPLE_RATE
    envelope = 0.20 + 0.65 * (t / seconds)
    calibration = envelope * (0.35 * np.sin(2 * np.pi * 220 * t))
    for onset in (0.5, 1.25, 2.0, 3.0):
        idx = int(onset * SAMPLE_RATE)
        u = np.arange(min(int(.18 * SAMPLE_RATE), len(t) - idx)) / SAMPLE_RATE
        calibration[idx:idx + len(u)] += .7 * np.sin(2 * np.pi * 70 * u) * np.exp(-22 * u)
    calibration *= TARGET_PEAK / (np.max(np.abs(calibration)) + 1e-9)
    cal_path = ROOT / "songs/calibration.wav"
    sf.write(cal_path, calibration.astype(np.float32), SAMPLE_RATE, subtype="PCM_16")
    print(f"wrote {cal_path.relative_to(ROOT)} (held-out synthetic calibration signal)")


if __name__ == "__main__":
    main()
