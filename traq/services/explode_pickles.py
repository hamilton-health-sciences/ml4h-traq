import multiprocessing as mp
import os
import pickle

from tqdm import tqdm

from traq.data import PickledStudy
from traq.utils import preprocess_plate


def explode_plate(plate):
    preprocessed = preprocess_plate(plate)
    if not preprocessed:
        return
    X, y = preprocessed

    return {plate.name: (X, y)}


def explode_trial(dataset, num_workers=48):
    """
    Params:
        dataset: The trial snapshot to compute metafeatures for.
    """
    results = {}
    with mp.Pool(num_workers) as pool:
        f = explode_plate
        plates = dataset.plates()
        for plate_results in tqdm(pool.imap_unordered(f, plates), total=len(plates)):
            if plate_results is not None:
                results = {**results, **plate_results}

    return results


def main(args):
    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    exploded = {}
    for snapshot_filename in tqdm(os.listdir(args.input_directory)):
        snapshot_filepath = os.path.join(args.input_directory, snapshot_filename)
        snapshot_name = snapshot_filename.split(".")[0]
        dataset = PickledStudy(snapshot_filepath)
        exploded[snapshot_name] = explode_trial(dataset)
    pickle.dump(exploded, open(args.output_filename, "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_directory", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    args = parser.parse_args()

    main(args)
