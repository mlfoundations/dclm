import json
import yaml
from datetime import datetime
import uuid
import os
import git
import boto3
import json
from cloudpathlib import S3Path
import glob
import pathlib

DATASET_REFS_DIR = os.path.join(
    pathlib.Path(__file__).parent.parent.absolute(),
    "exp_data",
    "datasets",
)


def get_source_ref_by_key(search_value, key="name", tokenized=False):
    for f in glob.glob(f"{DATASET_REFS_DIR}/*/*.json"):
        with open(f, "r") as file:
            ref = json.load(open(f, "r"))
            if ref.get(key, None) == search_value and ref.get("tokenized", None) == tokenized:
                return ref
    return None


def get_source_ref(source_ref_path):
    with open(source_ref_path, "r") as file:
        return json.load(file)


def count_tokens(manifest_url, seqlen=2049):
    if manifest_url.startswith("s3://"):
        manifest_url = S3Path(manifest_url).open("r")
    else:
        manifest_url = open(manifest_url, "r")
    
    with manifest_url as f:
        manifest = [json.loads(line) for line in f]
    num_tokens = sum(int(line["num_sequences"]) for line in manifest) * seqlen
    return num_tokens


def get_s3_dir_size(dataset_path):
    bucket, prefix = dataset_path.replace("s3://", "").split("/", 1)
    total_size = 0
    for i, obj in enumerate(boto3.resource("s3").Bucket(bucket).objects.filter(Prefix=prefix)):
        total_size += obj.size
    return total_size


def get_dir_size(dataset_path):
    if dataset_path.startswith("s3://"):
        return get_s3_dir_size(dataset_path)
    else:
        return None  # no count now


def get_git_info():
    repo = git.Repo(search_parent_directories=True)
    dcnlp_commit_hash = repo.head.object.hexsha
    dcnlp_diff = repo.git.diff(repo.head.commit.tree)
    return dcnlp_commit_hash, dcnlp_diff


def generate_untokenized_dataset_json(args, source_refs, base_output_path, data_key=".json.zstd"):
    sources = [{"uuid": s["uuid"], "name": s["name"]} for s in source_refs] if source_refs else []
    dcnlp_commit_hash, dcnlp_diff = get_git_info()

    dataset_json = {
        "uuid": str(uuid.uuid4().__str__()),
        "name": args.readable_name,
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "dataset_url": os.path.join(base_output_path, "processed_data/"),
        "manifest_url": None,
        "sources": sources,
        "tokenized": False,
        "tokenizer": None,
        "num_tokens": None,
        "size": get_dir_size(args.output_dir),
        "dcnlp_commit_hash": dcnlp_commit_hash,
        "dcnlp_diff": dcnlp_diff,
        "data_key": data_key,
    }

    return dataset_json


def generate_tokenized_dataset_json(args, source_refs, data_key="json.gz"):

    manifest_url = os.path.join(args.output.rstrip("/"), "manifest.jsonl")
    dcnlp_commit_hash, dcnlp_diff = get_git_info()
    sources = [{"uuid": s["uuid"], "name": s["name"]} for s in source_refs] if source_refs else []

    # TODO: Currently I just dump the entire yaml, is this the best thing to do?
    # Also, maybe would be nice to support automated generation of this yaml given input sources + weights
    sampling_yaml = None
    if args.do_sample:
        with open(args.default_dataset_yaml, "r") as file:
            sampling_yaml = yaml.safe_load(file)

    dataset_json = {
        "uuid": str(uuid.uuid4().__str__()),
        "name": args.readable_name,
        "creation_date": datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
        "dataset_url": args.output,
        "manifest_url": manifest_url,
        "sources": sources,
        "tokenized": True,
        "tokenizer": args.tokenizer,
        "num_tokens": count_tokens(manifest_url, args.seqlen + 1),
        "size": get_dir_size(args.output),
        "dcnlp_commit_hash": dcnlp_commit_hash,
        "dcnlp_diff": dcnlp_diff,
        "data_key": data_key,
        "sampling_yaml": sampling_yaml,
    }

    return dataset_json
