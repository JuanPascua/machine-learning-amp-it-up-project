import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path, set_seed
from src.data_pipeline import load_processed
from src.experiment_log import append_run
from src.metrics import report
from src.models import gain_from_train

#Runs the naive benchmarks
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    x_train, y_train = load_processed(config, "train")
    gain = gain_from_train(x_train, y_train)

    benchmarks_data = {"gain_matched_gain": gain, "naive_dry": {}, "naive_gain_matched": {}}
    for subset in ("train", "valid", "test"):
        x_subset, y_subset = load_processed(config, subset)
        benchmarks_data["naive_dry"][subset] = report(y_subset, x_subset)
        benchmarks_data["naive_gain_matched"][subset] = report(y_subset, gain * x_subset)

    results_dir = repo_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "benchmarks.json", "w") as file:
        json.dump(benchmarks_data, file, indent=2)

    for name in ("naive_dry", "naive_gain_matched"):
        append_run(config, name, valid_metrics=benchmarks_data[name]["valid"], test_metrics=benchmarks_data[name]["test"])
        test_metrics = benchmarks_data[name]["test"]
        print(f"{name}: test MSE={test_metrics['mse']:.6g}  ESR={test_metrics['esr']:.6g}")
    print(f"gain (train RMS wet / RMS dry) = {gain:.4f}")


if __name__ == "__main__":
    main()
