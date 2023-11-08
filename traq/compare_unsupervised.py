import os

import pandas as pd
from tqdm import tqdm

from traq.data import PickledStudy
from traq.evaluation import evaluate_pyod_models
from traq.grid import cash


def main(args):
    results = []
    for snapshot_filename in tqdm(os.listdir(args.input_directory)):
        snapshot_filepath = os.path.join(args.input_directory, snapshot_filename)
        trial_name = os.path.basename(args.input_directory)
        snapshot_name = snapshot_filename.split(".")[0]
        dataset = PickledStudy(snapshot_filepath)
        results_snapshot = evaluate_pyod_models(dataset, cash, trial_name)
        results_snapshot["snapshot"] = snapshot_name
        results.append(results_snapshot)
    results_df = pd.concat(results, axis=0)

    output_dirname = os.path.dirname(args.output_filename)
    os.makedirs(output_dirname, exist_ok=True)
    results_df.to_csv(args.output_filename)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    args = parser.parse_args()

    main(args)
