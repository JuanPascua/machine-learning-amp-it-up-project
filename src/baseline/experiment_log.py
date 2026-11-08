import datetime
import subprocess

from .config import REPO_ROOT

LOG_PATH = REPO_ROOT / "experiment_log.md"
HEADER = (
    "# Experiment log\n\n"
    "| Date (UTC) | Commit | Config | Seed | Run | Valid MSE | Valid ESR | Test MSE | Test ESR | Notes |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def _commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "no-git"


def format_value(value):
    return "" if value is None else f"{value:.6g}"

#Adds a log to logs of what was run and metrics.
def append_run(config, run_name, valid=None, test=None, notes="", valid_metrics=None, test_metrics=None):
    if not LOG_PATH.exists():
        LOG_PATH.write_text(HEADER)
    valid_metrics = valid_metrics if valid_metrics is not None else (valid or {})
    test_metrics = test_metrics if test_metrics is not None else (test or {})
    row = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
        _commit(), config["_config_path"], config["seed"], run_name,
        format_value(valid_metrics.get("mse")), format_value(valid_metrics.get("esr")),
        format_value(test_metrics.get("mse")), format_value(test_metrics.get("esr")), notes,
    )
    with open(LOG_PATH, "a") as file:
        file.write(row)
