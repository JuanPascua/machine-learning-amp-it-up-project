import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, repo_path, set_seed
from src.data_pipeline import load_processed
from src.experiment_log import append_run
from src.metrics import report
from src.models import accumulate_normal_equations, linear_predict, solve_ridge

# Trains the base line ridge regression model
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])
    num_past_samples_grid = sorted(int(n) for n in config["baseline"]["num_past_samples_grid"])
    alpha_grid = [float(a) for a in config["baseline"]["alpha_grid"]]

    x_train, y_train = load_processed(config, "train")
    x_valid, y_valid = load_processed(config, "valid")
    x_test, y_test = load_processed(config, "test")

    xtx, xty, count = accumulate_normal_equations(x_train, y_train, max(num_past_samples_grid))

    validation_grid, best_model = [], None
    #Using grid search to find the best lambda(penalty factor but is called alpha due to the python lambda variable name being for lambda functions and not usable)
    for num_past_samples in num_past_samples_grid:
        for alpha in alpha_grid:
            weights = solve_ridge(xtx, xty, count, num_past_samples, alpha)
            valid_metrics = report(y_valid, linear_predict(x_valid, weights))
            validation_grid.append({
                "num_past_samples": num_past_samples,
                "alpha": alpha,
                "valid_mse": valid_metrics["mse"],
                "valid_esr": valid_metrics["esr"],
            })
            if best_model is None or valid_metrics["mse"] < best_model["valid_metrics"]["mse"]:
                best_model = {
                    "num_past_samples": num_past_samples,
                    "alpha": alpha,
                    "valid_metrics": valid_metrics,
                    "weights": weights,
                }

    if best_model is None:
        raise RuntimeError("Grid search produced no models. Check num_past_samples_grid and alpha_grid in config.")

    train_metrics = report(y_train, linear_predict(x_train, best_model["weights"]))
    test_metrics = report(y_test, linear_predict(x_test, best_model["weights"]))

    results_dir = repo_path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "ridge_weights.npy", best_model["weights"])

    baseline_data = {
        "chosen": {
            "num_past_samples": best_model["num_past_samples"],
            "alpha": best_model["alpha"],
        },
        "train": train_metrics,
        "valid": best_model["valid_metrics"],
        "test": test_metrics,
        "validation_grid": validation_grid,
    }
    with open(results_dir / "baseline.json", "w") as file:
        json.dump(baseline_data, file, indent=2)

    append_run(config, f"ridge(N={best_model['num_past_samples']},alpha={best_model['alpha']})",
               valid_metrics=best_model["valid_metrics"], test_metrics=test_metrics)
    print(f"chosen N={best_model['num_past_samples']}, alpha={best_model['alpha']} (by validation MSE)")
    print(f"ridge: train MSE={train_metrics['mse']:.6g}  ESR={train_metrics['esr']:.6g}")
    print(f"ridge: test MSE={test_metrics['mse']:.6g}  ESR={test_metrics['esr']:.6g}")


if __name__ == "__main__":
    main()
