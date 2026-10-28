import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .config import repo_path, resolve_data_dir

# To find pairs that have both a dry_tag and wet_tag tagged file for the effect we are looking for in the subset we want (test/valid/train) 
def find_pairs(subset_dir, effect_filter, dry_tag, wet_tag):
    pairs = []
    for dry_path in sorted(Path(subset_dir).rglob("*.wav")):
        if dry_tag not in dry_path.name:
            continue
        if effect_filter and effect_filter.lower() not in str(dry_path).lower():
            continue
        wet_path = dry_path.with_name(dry_path.name.replace(dry_tag, wet_tag, 1))
        if wet_path.exists():
            pairs.append((dry_path, wet_path))
    # If no dry/wet pairs are found, stop execution with a clear error message.
    if not pairs:
        raise SystemExit(
            f"No dry/wet pairs found under {subset_dir} "
            f"(effect_filter={effect_filter!r}, dry_tag={dry_tag!r}, wet_tag={wet_tag!r}). "
            "Check the dataset layout and edit configs/baseline.yaml or find_pairs()."
        )
    return pairs

#Used to ensure that if a wav file is in stereo dual channel instead of momo that that it is monotized or if the sample rate is not what is expected.
def load_mono(path, expected_sr):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != expected_sr:
        raise SystemExit(f"{path} has sample rate {sr}, expected {expected_sr}.")
    return audio.mean(axis=1)

# Breaks down the wav files into smaller parts which are easier for processing
def trim_and_slice(x, y, trim_start, trim_end, slice_len):
    """Trim sync impulses, then cut into non-overlapping slices (remainder is dropped)."""
    min_len = min(len(x), len(y))
    end = min_len - trim_end if trim_end > 0 else min_len
    x, y = x[trim_start:end], y[trim_start:end]
    num_slices = len(x) // slice_len
    if num_slices == 0:
        return np.empty((0, slice_len), np.float32), np.empty((0, slice_len), np.float32)
    x = x[: num_slices * slice_len].reshape(num_slices, slice_len)
    y = y[: num_slices * slice_len].reshape(num_slices, slice_len)
    return x, y

#Prepares the subset train test or valid(ation) using the config path.
def prepare_subset(config, subset):
    data_config = config["data"]
    subsets = data_config["subsets"]
    subset_dir = resolve_data_dir(config) / subsets[subset]
    slice_len = int(round(data_config["slice_seconds"] * data_config["sample_rate"]))
    pairs = find_pairs(subset_dir, data_config["effect_filter"], data_config["dry_tag"], data_config["wet_tag"])

    x_temp, y_temp, file_indices = [], [], []
    for i, (dry_path, wet_path) in enumerate(pairs):
        x = load_mono(dry_path, data_config["sample_rate"])
        y = load_mono(wet_path, data_config["sample_rate"])
        x_slices, y_slices = trim_and_slice(
            x, y, int(data_config["trim_start_samples"]), int(data_config["trim_end_samples"]), slice_len
        )
        x_temp.append(x_slices)
        y_temp.append(y_slices)
        file_indices.extend([i] * len(x_slices))

    x_all = np.concatenate(x_temp, axis=0)
    y_all = np.concatenate(y_temp, axis=0)
    summary = {
        "subset": subset,
        "num_files": len(pairs),
        "num_slices": int(len(x_all)),
        "slice_seconds": data_config["slice_seconds"],
        "hours_kept": float(len(x_all) * data_config["slice_seconds"] / 3600.0),
        "file_stems": [pair[0].stem for pair in pairs],
    }
    return x_all, y_all, np.array(file_indices), summary


# Saves the processed data of the subset in compressed numpy file form where the processed data is located.
def save_processed(config, subset, x, y, file_indices):
    processed_dir = repo_path(config["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(processed_dir / f"{subset}.npz", x=x, y=y, file_idx=file_indices)

# Loads the processed data for a subset using its compressed numpy file from the processed data directory
def load_processed(config, subset):
    processed_file = repo_path(config["data"]["processed_dir"]) / f"{subset}.npz"
    if not processed_file.exists():
        raise SystemExit(f"{processed_file} not found. Run scripts/prepare.py first.")
    archive = np.load(processed_file)
    return archive["x"], archive["y"]

# Writes a summary for the data.
def write_data_summary(config, summaries):
    """Subset sizes and an overlap check, for the memo's subset documentation."""
    stems = {summary["subset"]: set(summary["file_stems"]) for summary in summaries}
    subset_names = list(stems)
    overlap = {
        f"{subset_a}&{subset_b}": sorted(stems[subset_a] & stems[subset_b])
        for i, subset_a in enumerate(subset_names) for subset_b in subset_names[i + 1:]
    }
    summary_report = {
        "subsets": [{k: v for k, v in summary.items() if k != "file_stems"} for summary in summaries],
        "file_name_overlap_between_subsets": overlap,
    }

    results_dir = repo_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "data_summary.json", "w") as file:
        json.dump(summary_report, file, indent=2)
    return summary_report
