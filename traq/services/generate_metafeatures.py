import multiprocessing as mp
import os
import pickle

from metaod.models.gen_meta_features import generate_meta_features
from tqdm import tqdm


def compute_metafeatures_plate(plate):
    plate_name, (X, _) = plate
    metafeatures = generate_meta_features(X)

    return {plate_name: metafeatures}


def compute_metafeatures(datasets, num_workers=48):
    results = {}
    with mp.Pool(num_workers) as pool:
        f = compute_metafeatures_plate
        for plate_results in tqdm(
            pool.imap_unordered(f, datasets.items()), total=len(datasets)
        ):
            if plate_results is not None:
                results = {**results, **plate_results}

    return results


def main(args):
    os.makedirs(os.path.dirname(args.output_filename), exist_ok=True)

    trial = pickle.load(open(args.input_filename, "rb"))

    metafeatures = {}
    for snapshot_name in tqdm(trial):
        metafeatures[snapshot_name] = compute_metafeatures(trial[snapshot_name])
    pickle.dump(metafeatures, open(args.output_filename, "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", type=str, required=True)
    parser.add_argument("--output_filename", type=str, required=True)
    args = parser.parse_args()

    main(args)
