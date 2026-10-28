
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path


def format_value(value):
    return "" if value is None else f"{value:.6g}"

#Saves the results of the benchmarks and baseline into as a csv and markdown file where the weights for the ridge regression model is.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = repo_path(config["results_dir"])
    with open(results_dir / "benchmarks.json", "r") as file:
        benchmarks_data = json.load(file)
    with open(results_dir / "baseline.json", "r") as file:
        baseline_data = json.load(file)
    expected_results = config.get("expected_results") or {}

    chosen_samples = baseline_data["chosen"]["num_past_samples"]
    ridge_name = f"Ridge regression (N={chosen_samples}, alpha={baseline_data['chosen']['alpha']})"
    rows = [
        ("Naive benchmark (dry = prediction)", "naive_dry",
         benchmarks_data["naive_dry"].get("train"), benchmarks_data["naive_dry"]["valid"], benchmarks_data["naive_dry"]["test"]),
        ("Naive benchmark (gain-matched)", "naive_gain_matched",
         benchmarks_data["naive_gain_matched"].get("train"), benchmarks_data["naive_gain_matched"]["valid"], benchmarks_data["naive_gain_matched"]["test"]),
        (ridge_name, "ridge",
         baseline_data.get("train"), baseline_data["valid"], baseline_data["test"]),
    ]

    has_expected = any(
        (expected_results.get(key) or {}).get("mse") is not None
        for _, key, _, _, _ in rows
    )

    header = ["Model", "Train MSE", "Train ESR", "Valid MSE", "Valid ESR", "Test MSE", "Test ESR"]
    if has_expected:
        header += ["Expected MSE", "Expected ESR", "MSE difference", "ESR difference"]

    table = []
    for label, key, train_metrics, valid_metrics, test_metrics in rows:
        row = [
            label,
            format_value(train_metrics.get("mse") if train_metrics else None),
            format_value(train_metrics.get("esr") if train_metrics else None),
            format_value(valid_metrics["mse"]),
            format_value(valid_metrics["esr"]),
            format_value(test_metrics["mse"]),
            format_value(test_metrics["esr"]),
        ]
        if has_expected:
            expected_metrics = expected_results.get(key) or {}
            expected_mse = expected_metrics.get("mse")
            expected_esr = expected_metrics.get("esr")
            diff_mse = None if expected_mse is None else test_metrics["mse"] - expected_mse
            diff_esr = None if expected_esr is None else test_metrics["esr"] - expected_esr
            row.extend([
                format_value(expected_mse),
                format_value(expected_esr),
                format_value(diff_mse),
                format_value(diff_esr),
            ])
        table.append(row)

    with open(results_dir / "results_table.csv", "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(header)
        csv_writer.writerows(table)

    markdown_lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    markdown_lines += ["| " + " | ".join(row) + " |" for row in table]
    if expected_results.get("source"):
        markdown_lines.append(f"\nExpected values source: {expected_results['source']}")
    (results_dir / "results_table.md").write_text("\n".join(markdown_lines) + "\n")
    print("\n".join(markdown_lines))
    print(f"\nWrote {results_dir / 'results_table.csv'} and {results_dir / 'results_table.md'}")


if __name__ == "__main__":
    main()
