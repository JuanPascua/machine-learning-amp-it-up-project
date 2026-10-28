
import argparse
import sys
from pathlib import Path

#Prepares the dataset so that we can train our model on good clean data rather than messy unclean or unoptimized data forms.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, set_seed
from src.data_pipeline import prepare_subset, save_processed, write_data_summary



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"])

    subsets = config["data"]["subsets"]
    summaries = []
    for subset in subsets:
        x_all, y_all, file_indices, summary = prepare_subset(config, subset)
        save_processed(config, subset, x_all, y_all, file_indices)
        summaries.append(summary)
        print(f"{subset}: {summary['num_files']} files, {summary['num_slices']} slices, "
              f"{summary['hours_kept']:.3f} h kept")

    summary_report = write_data_summary(config, summaries)
    overlap = summary_report["file_name_overlap_between_subsets"]
    for pair, names in overlap.items():
        status = "OK (none)" if not names else f"WARNING: {len(names)} shared names"
        print(f"file-name overlap {pair}: {status}")


if __name__ == "__main__":
    main()
