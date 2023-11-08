import os
from collections import OrderedDict

import pandas as pd
from pyod.models.iforest import IForest
from tqdm import tqdm

from traq.data import PickledStudy
from traq.meta_evaluation import evaluate_model_selection
from traq.metaod_wrapper import Ensemble, MetaODPE, MetaODWrapper


def main(args):
    results = []
    for snapshot_filename in tqdm(os.listdir(args.input_directory)):
        snapshot_filepath = os.path.join(args.input_directory, snapshot_filename)
        trial_name = os.path.basename(args.input_directory)
        snapshot_name = snapshot_filename.split(".")[0]
        dataset = PickledStudy(snapshot_filepath)
        cash = OrderedDict(
            {
                IForest: OrderedDict({}),  # global best
                MetaODWrapper: OrderedDict(
                    {"model_name": ["default", f"{trial_name}.pkl"]}
                ),
                Ensemble: OrderedDict({"transform": [None, "probabilistic"]}),
                MetaODPE: OrderedDict({"model_name": [f"{trial_name}.pkl"], "k": [3]}),
            }
        )
        results_snapshot = evaluate_model_selection(dataset, cash, trial_name)
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
