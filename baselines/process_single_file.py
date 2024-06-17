import argparse

from core import process_single_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", required=True, help="Path to the YAML file.")
    parser.add_argument("--raw_data_dirpath", required=True, help="he path to the top data directory in the data "
                                                                  "hierarchy (from which to mirror the output path)")
    parser.add_argument("--jsonl", required=True, help="path to the input JSONL file with the text to be processed, "
                                                       "relative to raw_data_dirpath")
    parser.add_argument("--source_name", required=True, help="The name of the source of the jsonl file.")
    parser.add_argument("--output_dir", required=True, help="Path to the output dir of the processed file.")
    parser.add_argument("--workers", type=int, required=False, default=1,
                        help="If larger than one, will use a process pool with that many workers.")
    parser.add_argument("--overwrite", action='store_true', required=False,
                        help="If set to true, will overwrite results.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_single_file(config_path=args.yaml,
                        raw_data_dirpath=args.raw_data_dirpath,
                        jsonl_relpath=args.jsonl,
                        source_name=args.source_name,
                        output_dir=args.output_dir,
                        workers=args.workers,
                        overwrite=args.overwrite)
