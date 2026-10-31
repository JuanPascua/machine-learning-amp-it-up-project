# Amp it Up: Baseline Reproduction (overdrive effect, ToneTwist AFx)

## Purpose
Reproducible baseline for modeling a guitar-amp **overdrive** effect (dry input -> wet output).
It contains the data pipeline, two naive benchmarks (dry = prediction, gain-matched dry),
and a course-level baseline (ridge regression = a linear model on the past N dry samples).
Main metric: MSE. Secondary: ESR (MSE divided by target energy). Lower is better for both.

## Repository structure
```
README.md
requirements.txt          pinned package versions
configs/baseline.yaml     all hyperparameters for the real run (idea for usinf a config file came from the config mention in the example)
configs/sample.yaml       same pipeline on the tiny synthetic sample
src/                      config, data_pipeline, models, metrics, experiment_log
scripts/                  prepare, run_benchmarks, train_baseline, make_table, make_sample_data
sample_data/              tiny SYNTHETIC dry/wet clips for a quick check (not ToneTwist data)
experiment_log.md         one row per run (appended automatically)
results/                  generated tables and JSON (not committed)
```

## Setup
```
python -m venv .venv && source .venv/bin/activate or ./venv/Scripts/activate
pip install -r requirements.txt 
```
## Errors During Setup:
You might get package not found errors being shown when looking at the python files or when attempting to run the files. If this happens to you ensure your using the .venv's python interpretor 
rather then you global or other python interpretors.
If you encounter other possible problems please install and use python 3.12 as we used it for the training and preperation of the model and have encountered no problems.

## How to get the full Dataset (Ignore this we have decided to use one dataset for now and update the repo to make your grading easier for milestone 3)
The full dataset is large so only a tiny sample is included in the repo and some local setup is required. It will be ammended or removed in a later milestone depending on the direction the project takes.
1. Download ToneTwist AFx data from its github repo at  https://github.com/mcomunita/tonetwist-afx-dataset/tree/master follow their instructions to download it.
2. Put the provided `train/`, `val/`, `test/` folders into a single folder and then create an environment varaiable that is used by the config file to get the data and prepare it:
   `export DATA_DIR=/path/to/tonetwist_afx` (Windows PowerShell: `$env:DATA_DIR="C:\path\to\tonetwist_afx"`).
3. The subsets are used exactly as provided. `scripts/prepare.py` reports subset sizes and checks that no file name appears in two subsets (`results/data_summary.json`).

## The data is now located in the data folder from which you can get an actual baseline. (Important Note)
When we submitted our milestone 3 we were having problems with using google collab where downloading data from the afxtonetwistafx repo was taking forever so we provided and used those very samples. Today we had enough and just tried to run it locally and it worked so the 
github is getting this commit and well update. Please don't add an additional late penalty for this but this now include data so you don't have to install it and the results of the benchmarks. Their is also comment in canvas to inform you of this.

## Data Layout assumption. 
`src/data_pipeline.py::find_pairs` expects each dry file and its wet file in the same folder, with names that differ only by the tag in `dry_tag` / `wet_tag` (default `dry` / `wet`), and the text `overdrive` somewhere in the path. If the real dataset is organised differently, edit `find_pairs` or the three keys in `configs/baseline.yaml`. This is the one part that must be checked against the real data.


## Reproduce the main result by executing the files in the following order doing it another issues can and will result in issues.
```
python scripts/prepare.py        --config configs/baseline.yaml
python scripts/run_benchmarks.py --config configs/baseline.yaml
python scripts/train_baseline.py --config configs/baseline.yaml
python scripts/make_table.py     --config configs/baseline.yaml
```
Output: `results/results_table.csv` and `results/results_table.md` (the baseline and benchmarks table for the memo),
plus `results/baseline.json` (chosen N and alpha, full validation grid), `results/benchmarks.json`,
`results/data_summary.json`, and `results/ridge_weights.npy`.

## Quick check on the synthetic sample (about a minute, no download)
```
python scripts/make_sample_data.py     # only needed if sample_data/ is missing
python scripts/prepare.py        --config configs/sample.yaml
python scripts/run_benchmarks.py --config configs/sample.yaml
python scripts/train_baseline.py --config configs/sample.yaml
python scripts/make_table.py     --config configs/sample.yaml
```
Results go to `results/sample/`. These numbers only show the pipeline runs; they are not the reported results.

## Reproducibility details
- WE used seed number 42, set at the start of every script (`random`, NumPy, `PYTHONHASHSEED`). The ridge solution is closed-form, so results are deterministic for the same package versions.
- hyperparameters are set only in the config files. N and alpha are chosen by validation MSE as taught in class, and never on the test set.
- Gain-matched naive benchmark: The naive benchmarks gain is computed on the train subset and applied to valid/test (no test information used). State this in the memo.
- Metrics are grouped over all samples of all slices or parts of audio in a subset.
- Ridge alpha: Applied to the covariance normalised by sample count, so it is relative to the mean input power. No intercept is fitted.
- Experiment log: each script appends a log to  `experiment_log.md` (date, commit, config, seed, metrics).
- No secrets or personal paths are stored in the repository; the data path comes from `DATA_DIR`.


