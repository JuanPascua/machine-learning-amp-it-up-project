"""Generate a tiny SYNTHETIC dry/wet dataset in sample_data/ for a quick end-to-end check.

The 'wet' signal is a tanh soft-clipper applied to the dry signal (an overdrive-like
distortion). This is NOT ToneTwist data; numbers from it are not the reported results.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import REPO_ROOT

SR = 48000
SECONDS = 2.0
COUNTS = {"train": 3, "valid": 1, "test": 1}


def make_dry(rng, num_samples):
    time_axis = np.arange(num_samples) / SR
    signal = np.zeros(num_samples)
    for _ in range(4):
        freq = rng.uniform(80, 1200)
        signal += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * freq * time_axis + rng.uniform(0, 6.28))
    signal *= 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.5, 2.0) * time_axis) ** 2   # amplitude envelope
    signal += 0.01 * rng.standard_normal(num_samples)
    return (0.5 * signal / np.max(np.abs(signal))).astype(np.float32)


def main():
    rng = np.random.default_rng(42)
    num_samples = int(SR * SECONDS)
    for subset, count in COUNTS.items():
        subset_dir = REPO_ROOT / "sample_data" / subset
        subset_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            dry = make_dry(rng, num_samples)
            wet = np.tanh(6.0 * dry)
            wet = (0.5 * wet / np.max(np.abs(wet))).astype(np.float32)
            sf.write(subset_dir / f"sample_overdrive_{subset}_{i:02d}_dry.wav", dry, SR, subtype="PCM_16")
            sf.write(subset_dir / f"sample_overdrive_{subset}_{i:02d}_wet.wav", wet, SR, subtype="PCM_16")
    print("Wrote synthetic sample data to sample_data/")


if __name__ == "__main__":
    main()
