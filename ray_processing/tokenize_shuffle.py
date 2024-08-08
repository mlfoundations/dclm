import argparse
import os
from typing import List
import pathlib
import json

from utils import generate_tokenized_dataset_json, get_source_ref, get_source_ref_by_key
from training.dataset_reference import replace_prefix
from open_lm.datapreprocess.ray import tokenize_shuffle

DIR = pathlib.Path(__file__).parent.absolute()


def add_tokenize_shuffle_args(parser):
    # Args to be fed into tokenize_shuffle
    parser.add_argument("--input", help="input path", type=str)
    parser.add_argument("--output", help="output path", type=str, required=True)
    parser.add_argument("--content_key", type=str, default="text")
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--wds_chunk_size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--ray_address", type=str, default=None)
    parser.add_argument("--force_parallelism", type=int, default=None)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument(
        "--default_dataset_yaml", type=str, default=(DIR / "tokenization_configs" / "rpj_lm_data.yaml").__str__()
    )
    parser.add_argument("--num_writers_per_node", type=int, default=1)
    parser.add_argument("--ray_spill_location", type=str, default="/tmp/ray")
    parser.add_argument("--mirror", help="Use this dataset mirror if it exists in the dataset 'mirrors' key.")
    parser.add_argument(
        "--suffixes", nargs="+", default=[".jsonl", ".jsonl.gz", ".jsonl.zst", ".jsonl.zstd", ".tar", ".tar.gz"]
    )

    # Args specific to dcnlp pipeline (as opposed to tokenize_shuffle)
    DCNLP_ARGS = [
        "source_ref_paths",
        "readable_name",
        "overwrite",
        "do_sample",
        "no_shuffle",
        "prefix_replacement",
        "mirror",
    ]
    parser.add_argument(
        "--source_ref_paths", help="paths to untokenized datasets refs, comma or space separated", type=str, nargs="+"
    )
    parser.add_argument(
        "--readable_name", help="name given to tokenized dataset and reference json file name", type=str, required=True
    )
    parser.add_argument(
        "--overwrite", help="allow for overwriting the reference json, to be used sparingly", action="store_true"
    )
    parser.add_argument("--prefix_replacement", default="", help="Prefix replacement in S3 URL")
    return parser, DCNLP_ARGS


def main(args, dcnlp_arg_names):
    # Before proceeding with tokenization, make sure that an existing json won't be overwritten
    json_path = f"exp_data/datasets/tokenized/{args.readable_name}.json"
    if not args.overwrite:
        assert not os.path.exists(
            json_path
        ), f"{json_path} already exists. Try changing --readable_name or deleting the existing file."

    # Collect the dataset urls from the source reference json paths, if no explicity jsons provided, tries to search for them based on --input
    source_refs = None
    if args.source_ref_paths is not None:
        source_ref_paths = [p.strip() for paths in args.source_ref_paths for p in paths.split(",") if p.strip()]
        source_refs = [get_source_ref(s) for s in source_ref_paths]
        if args.mirror:
            for s in source_refs:
                if args.mirror in s.get("mirror", {}):
                    new_url = s["mirror"]["dataset_url"]
                    print(f"Updating dataset url for {s['name']}: {s['dataset_url']} -> {new_url}.")
                    s["dataset_url"] = new_url
        args.input = ",".join([replace_prefix(s["dataset_url"], args.prefix_replacement) for s in source_refs])

    # Collect args for tokenization and pass them into tokenize_shuffle
    tokenize_shuffle_args = [
        str(i)
        for k, v in vars(args).items()
        for i in [f"--{k}", v]
        if k not in dcnlp_arg_names and v and k != "suffixes"
    ]

    tokenize_shuffle_args.append("--suffixes")
    for suffix in args.suffixes:
        tokenize_shuffle_args.append(str(suffix))

    if args.do_sample:
        tokenize_shuffle_args.append("--do_sample")

    if args.no_shuffle:
        tokenize_shuffle_args.append("--no_shuffle")

    tokenize_shuffle.main(tokenize_shuffle_args)

    dataset_json = generate_tokenized_dataset_json(args, source_refs)
    with open(json_path, "w") as ref_file:
        json.dump(dataset_json, ref_file, indent=4)
    out_json_path = f"{args.output}/{pathlib.Path(args.output).name}.json"
    print(f"moving dataset json to {out_json_path}")
    if out_json_path.startswith("s3://"):
        os.system(f"aws s3 cp {json_path} {out_json_path}")
    else:
        os.system(f"mv {json_path} {out_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser, DCNLP_ARGS = add_tokenize_shuffle_args(parser)
    args = parser.parse_args()
    main(args, DCNLP_ARGS)
