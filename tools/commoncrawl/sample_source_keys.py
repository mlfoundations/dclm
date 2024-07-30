"""
Sample keys from an S3 bucket and generate a dataset JSON file.
"""

import argparse
import gzip
import io
import json
import pathlib
import random
import re
import subprocess
import time
import uuid

import boto3
import fsspec
from gen_common_crawl_paths import APPROX_TOKENS_PER_WARC
from tqdm import tqdm


def read_keys_from_file(file_path, regex_pattern):
    try:
        compiled_pattern = re.compile(regex_pattern)
    except re.error as e:
        print(f"Error compiling regex pattern: {e}")
        return []
    keys = []
    with fsspec.open(file_path, "rb") as file:
        if file_path.endswith(".gz"):
            with gzip.GzipFile(fileobj=io.BytesIO(file.read())) as gz:
                data = gz.read().decode("utf-8").splitlines()
        else:
            data = file.read().decode("utf-8").splitlines()

        for line in tqdm(data):
            key = line.strip()
            if compiled_pattern.match(key):
                keys.append(key)
    return keys


def list_matching_s3_keys(bucket, prefix="", regex_pattern=""):
    """List keys in an S3 bucket that match the given regex pattern."""
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in tqdm(page.get("Contents", [])):
            key = obj["Key"]
            if re.match(regex_pattern, key):
                yield key


def sample_keys(keys, subset_size, seed):
    """Randomly sample a subset of keys."""
    random.seed(seed)
    return random.sample(keys, min(subset_size, len(keys)))


def get_git_info():
    """Get the current git commit hash and diff."""
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
    git_diff = subprocess.check_output(["git", "diff"]).strip().decode()
    return commit_hash, git_diff


def generate_dataset_json(sampled_keys, name, file_path, bucket, prefix, seed):
    """Generate dataset JSON with mandatory keys."""
    if file_path:
        source = file_path
    else:
        source = f"{bucket}/{prefix}"
    git_info = get_git_info()
    dataset = {
        "uuid": str(uuid.uuid4()),
        "name": name,
        "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sources": source,
        "tokenized": False,
        "tokenizer": None,
        "size": sum([len(k) for k in sampled_keys]),  # Simplified size estimation
        "seed": seed,
        "dataset_urls": sampled_keys,
        "dcnlp_commit_hash": git_info[0],
        "dcnlp_diff": git_info[1],
    }
    return dataset


def main(
    file_path, bucket, prefix, regex_pattern, seed, subset_size, name, exp_data_path
):  # pylint: disable=too-many-arguments
    """
    Main function to generate dataset json
    """
    if file_path:
        all_keys = read_keys_from_file(file_path, regex_pattern)
    else:
        all_keys = list(list_matching_s3_keys(bucket, prefix, regex_pattern))

    sampled_keys = sample_keys(all_keys, subset_size, seed)
    dataset_json = generate_dataset_json(sampled_keys, name, file_path, bucket, prefix, seed)

    out_path = f"{exp_data_path}/datasets/{dataset_json['name']}.json.gz"
    assert pathlib.Path(out_path).exists() is False, f"dataset {out_path} already exists"
    dataset_json_compressed = gzip.compress(json.dumps(dataset_json, indent=4).encode())
    with open(
        out_path,
        "wb",
    ) as f_obj:
        f_obj.write(dataset_json_compressed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample keys from an S3 bucket and generate a dataset JSON file.")
    parser.add_argument("--bucket", default=None, help="S3 bucket name")
    parser.add_argument(
        "--keys-path",
        help="File path (local or S3 URL) containing the keys",
        default="s3://***REMOVED***/commoncrawl_paths.txt.gz",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Prefix for S3 bucket keys",
    )
    parser.add_argument("--regex", default="^(?!.*CC-MAIN-2023).*warc.*", help="Regex pattern to match S3 keys")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=int(200e12 // APPROX_TOKENS_PER_WARC),
        help="Size of the subset to sample (defaults to ~1e12 tokens worth of warcs)",
    )
    parser.add_argument("--name", required=True, help="Readable name for the dataset")
    parser.add_argument("--exp_data_path", help="path to exp data", default="exp_data")

    args = parser.parse_args()
    if (args.keys_path is None) == (args.bucket is None or args.prefix is None):
        parser.error("Either --keys-path or both --bucket and --prefix must be provided, but not both.")

    main(
        args.keys_path,
        args.bucket,
        args.prefix,
        args.regex,
        args.seed,
        args.subset_size,
        args.name,
        args.exp_data_path,
    )
