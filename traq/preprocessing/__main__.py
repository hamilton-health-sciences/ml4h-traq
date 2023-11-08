from .derive_labels import derive_labels
from .diff import diff
from .summarize import summarize


def main(args):
    if args.command == "diff":
        diff(args.config_filename, args.output_directory, args.num_workers)
    elif args.command == "summarize":
        summarize(args.config_filename, args.output_directory)
    elif args.command == "derive_labels":
        derive_labels(args.config_filename, args.output_directory)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", type=str, choices=["diff", "summarize", "derive_labels"]
    )
    parser.add_argument("--config_filename", type=str, required=True)
    parser.add_argument("--output_directory", type=str, default="output")
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    main(args)
