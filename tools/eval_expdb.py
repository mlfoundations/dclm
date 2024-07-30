"""
Functions:
- load_models: Loads models' information from a JSON file in a database, based on specified index range. Supports basic filtering.
- replace_prefix: Alters the prefix in an S3 URL, used for handling different storage locations.
- download_from_s3: Downloads files from an S3 URL, used for retrieving model checkpoints and parameters.
- download_checkpoint: Retrieves the model's checkpoint file from its URL.
- download_params: Retrieves the model's parameter file from its URL.
- run_eval: Executes an evaluation script for a model, using its checkpoint and parameter files.
- modify_and_save_evaluation_json: Stores the evaluation results in a JSON file, appending the model's UUID.
- check_path_exists: Verifies the existence of a given path in the filesystem, supporting both S3 and local paths.

Main Function:
Handles the process of loading, downloading, and evaluating models from a database.
- Extracts models from a Git-hosted database, applying optional filters.
- Downloads the necessary files (checkpoint and parameters) for each model.
- Runs evaluation scripts for each model.
- Saves evaluation results with model identification (UUID) to a specified output directory.

Key Requirements for Model Evaluation:
Each model in the database must have the following keys in its JSON representation:
- "checkpoint_url": URL to the model's checkpoint file.
- "params_url": URL to the model's parameter file.
- "model_uuid": Unique identifier for the model.

Parameters:
- database_path: Path to the model database.
- filters: Filters to apply when selecting models from the database.
- start_idx, end_idx: Index range for model selection.
- output_dir: Directory to store downloaded and evaluation files.
- compute_perplexity, hf_model, hf_cache_dir: Optional parameters for advanced model evaluation features.
- prefix_replacement: For modifying S3 URL prefixes.
- num_gpus: Number of GPUs to use for evaluation.
- eval_yaml: Path to the YAML configuration file for evaluation.
- eval_dir: Temporary directory for evaluation files.
- no_skip: Flag to override skipping of existing evaluations.

Usage:
Run the script with command-line options to perform model evaluations. For example:
`python script.py --database_path 'path/to/database' --start_idx 0 --end_idx 10 --output_dir 'path/to/output'`

Note:
- Requires AWS credentials for accessing S3.
- Assumes specific JSON format and directory structure as per the DataComp Language Model project.
"""

import json
import os
import pathlib
import shutil
import subprocess

import boto3
import click
import fsspec
import pandas as pd
from botocore.exceptions import NoCredentialsError
from tools.expdb import build_table_dfs, filter_df, merge_uuid_references
from loguru import logger


def tri_copy_model_via_hop(src, dst, profile):
    if profile in [None, ""]:
        profile_arg = ""
    else:
        profile_arg = f"--profile {profile}"
    src_split = src.split("/")
    name_idx = -1
    if src_split[name_idx].startswith("epoch_") or src_split[name_idx].startswith("params"):
        name_idx -= 1
    if src_split[name_idx].startswith("checkpoints"):
        name_idx -= 1
    model_name = f"{src_split[name_idx]}_{src_split[-1]}"
    if dst[-1] != "/":
        dst += "/"
    if src.startswith("s3://"):
        # Test if f"{dst}{model_name}" exists on s3
        list_file = subprocess.call(f"aws s3 ls {dst}{model_name}", shell=True)
        if list_file == 0:
            print(f"{dst}{model_name} already exists, no need to copy.")
            return f"{dst}{model_name}"

        if src.split("/")[2] == "dcnlp-west":
            print("Copying from dcnlp-west to dcnlp-east")
            if os.getenv("AWS_DCNLP_ACCESS_KEY_ID") is None:
                print("Trying to use dcnlp-west profile, it should be defined in your ~/.aws/config file")
                os.system(f"aws s3 cp {src} s3://***REMOVED***/tri-tmp/model/{model_name} --profile dcnlp-west")
            else:
                print("Using env variables for dcnlp-west")
                access_key = os.getenv("AWS_DCNLP_ACCESS_KEY_ID")
                secret_key = os.getenv("AWS_DCNLP_SECRET_ACCESS_KEY")
                os.system(
                    f"AWS_ACCESS_KEY_ID={access_key} AWS_SECRET_ACCESS_KEY={secret_key} aws s3 cp {src} s3://***REMOVED***/tri-tmp/model/{model_name}"
                )
            print("Copying from dcnlp-east to tmp-lm-data")
            os.system(
                f"aws {profile_arg} s3 cp s3://***REMOVED***/tri-tmp/model/{model_name} s3://tmp-lm-data/copy-data/model/{model_name}"
            )
            os.system(f"aws {profile_arg} s3 rm s3://***REMOVED***/tri-tmp/model/{model_name}")
            print("Copying from tmp-lm-data to destination")
            os.system(f"aws {profile_arg} s3 cp s3://tmp-lm-data/copy-data/model/{model_name} {dst}{model_name}")
            os.system(f"aws {profile_arg} s3 rm s3://tmp-lm-data/copy-data/model/{model_name}")
        elif src.split("/")[2] == "***REMOVED***":
            print("Copying from dcnlp-east to tmp-lm-data")
            os.system(f"aws {profile_arg} s3 cp {src} s3://tmp-lm-data/copy-data/model/{model_name}")
            print("Copying from tmp-lm-data to destination")
            os.system(f"aws {profile_arg} s3 cp s3://tmp-lm-data/copy-data/model/{model_name} {dst}{model_name}")
            os.system(f"aws {profile_arg} s3 rm s3://tmp-lm-data/copy-data/model/{model_name}")
        elif src.split("/")[2] == "***REMOVED***":
            return src
        else:
            os.system(f"aws {profile_arg} s3 cp {src} {dst}{model_name}")
    print(f"Copied model from {src} to {dst}")
    return f"{dst}{model_name}"


def load_models(database_path, table, start_idx, end_idx, filters):
    table_dfs = build_table_dfs(database_path, table)
    merged_dfs = merge_uuid_references(table_dfs)

    df = merged_dfs[table]  # Assuming 'models' is the table name

    # Apply filters to the DataFrame
    if filters:
        df = filter_df(df, filters)

    print(df)
    return df.iloc[start_idx:end_idx]


def replace_prefix(s3_url, prefix_replacement):
    old_prefix, new_prefix = prefix_replacement.split("=")
    if s3_url.startswith(old_prefix):
        return s3_url.replace(old_prefix, new_prefix, 1)
    return s3_url


def download_from_s3(s3_url, output_dir, prefix_replacement=None, profile=None):
    if prefix_replacement:
        s3_url = replace_prefix(s3_url, prefix_replacement)
    if profile is not None:
        profile = f"--profile {profile}"
    else:
        profile = ""

    try:
        local_filename = os.path.join(output_dir, s3_url.split("/")[-1])
        print(f"Downloading {s3_url} to {local_filename}")
        os.system(f"aws s3 cp {s3_url} {local_filename} {profile}")
        return local_filename
    except NoCredentialsError:
        print("Credentials not available for AWS S3.")
        return None


def download_checkpoint(
    model_row,
    output_dir,
    prefix_replacement,
    tri_s3_path=None,
    local_download=True,
    profile=None,
    checkpoint_replacement=None,
):
    checkpoint_url = model_row["checkpoint_url"]

    if checkpoint_replacement is not None:
        checkpoint_dir = os.path.split(checkpoint_url)[0]
        checkpoint_url = os.path.join(checkpoint_dir, checkpoint_replacement)

    if tri_s3_path is not None:
        checkpoint_url = tri_copy_model_via_hop(checkpoint_url, tri_s3_path, profile=profile)

    if local_download and checkpoint_url.startswith("s3://"):
        return download_from_s3(checkpoint_url, output_dir, prefix_replacement, profile=profile)
    else:
        return checkpoint_url


def download_params(model_row, output_dir, prefix_replacement, tri_s3_path=None, local_download=True, profile=None):
    if tri_s3_path is not None:
        params_url = tri_copy_model_via_hop(model_row["params_url"], tri_s3_path, profile=profile)
    else:
        params_url = model_row["params_url"]
    if local_download and params_url.startswith("s3://"):
        return download_from_s3(params_url, output_dir, prefix_replacement, profile=profile)
    else:
        return params_url


def run_eval(
    eval_script,
    model_checkpoint,
    tokenizer,
    model_config,
    eval_yaml,
    params_file,
    output_dir,
    skip_perplexity,
    averager_name,
    hf_model,
    hf_cache_dir,
    num_gpus,
    force_xformers,
):
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(num_gpus),
        eval_script,
        "--checkpoint",
        model_checkpoint,
        "--tokenizer",
        tokenizer,
        "--eval-yaml",
        eval_yaml,
        "--config",
        params_file,
        "--model",
        model_config,
        "--output-file",
        "eval_output.json",
    ]
    if averager_name:
        cmd.extend(["--averager-name", averager_name])
    if skip_perplexity:
        cmd.append("--donot-compute-perplexity")
    else:
        cmd.append("--compute-downstream-perplexity")

    if hf_model:
        cmd.extend(["--hf-model", hf_model])
    if hf_cache_dir:
        cmd.extend(["--hf-cache-dir", hf_cache_dir])

    if force_xformers:
        cmd.extend(["--force-xformers"])

    print(f"Running cmd:\n{cmd}")
    subprocess.run(cmd, check=True)
    with open("eval_output.json") as f:
        return json.load(f)


def modify_and_save_evaluation_json(eval_data, model_uuid, output_dir, evaluation_name):
    eval_data["model_uuid"] = model_uuid
    destination = os.path.join(output_dir, evaluation_name)
    print(f"Saving json to {destination}")
    with fsspec.open(destination, "w") as f:
        json.dump(eval_data, f, indent=4)


def check_path_exists(path):
    # Determine the file system type based on the path prefix
    if path.startswith("s3://"):
        # S3 file system
        # Note: You need to have s3fs installed and AWS credentials set up
        fs = fsspec.filesystem("s3", anon=False)
    else:
        # Local file system
        fs = fsspec.filesystem("file")

    # Check if the path exists
    return fs.exists(path)


@click.command()
@click.option("--database_path", default="exp_data", help="Path to the database")
@click.option("--tri_s3_path", default=None, help="S3 path, used only for TRI")
@click.option("--table", default="models", help="models table")
@click.option("--filters", "-f", multiple=True, help="Filters to apply on the models table")
@click.option("--start_idx", default=0, type=int, help="Start index for model filtering")
@click.option("--end_idx", default=10, type=int, help="End index for model filtering")
@click.option("--output_dir", default=".", help="Directory to save checkpoints and evaluations")
@click.option("--skip_perplexity", is_flag=True, help="skip perplexity evaluation")
@click.option("--averager_name", default=None)
@click.option("--hf_model", default=None, help="HF model name for evaluation")
@click.option("--hf_cache_dir", default=None, help="Custom cache directory for HF models")
@click.option("--prefix_replacement", default=None, help="Prefix replacement in S3 URL")
@click.option("--num_gpus", default=1, type=int, help="Number of GPUs to use")
@click.option("--eval_yaml", default="eval/light.yaml", type=str, help="which eval yaml to use")
@click.option("--eval_dir", default="/tmp/dcnlp_eval/", type=str, help="which eval yaml to use")
@click.option("--tokenizer", default="EleutherAI/gpt-neox-20b", help="tokenizer")
@click.option("--no_skip", is_flag=True, help="do not skip evals if they exist")
@click.option("--profile", default=None, help="AWS profile to use")
@click.option("--force_xformers", is_flag=True, help="Force xformers attention")
@click.option("--checkpoint_replacement", type=str, default=None, help="Checkpoint name to evaluate at.")
def main(
    database_path,
    tri_s3_path,
    table,
    filters,
    start_idx,
    end_idx,
    output_dir,
    skip_perplexity,
    averager_name,
    hf_model,
    hf_cache_dir,
    prefix_replacement,
    num_gpus,
    eval_yaml,
    eval_dir,
    tokenizer,
    no_skip,
    profile,
    force_xformers,
    checkpoint_replacement,
):
    CWD = os.getcwd()
    if not output_dir.startswith("s3://") and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir, exist_ok=False)
    database_path = f"{CWD}/{database_path}"
    eval_yaml = f"{CWD}/{eval_yaml}"
    eval_type = pathlib.Path(eval_yaml).stem
    models_df = load_models(database_path, table, start_idx, end_idx, filters)

    for _, model_row in models_df.iterrows():
        if checkpoint_replacement is not None:
            eval_name = f"evaluation_{model_row['name']}_{checkpoint_replacement}_{eval_type}.json"
        else:
            eval_name = f"evaluation_{model_row['name']}_{eval_type}.json"

        destination = os.path.join(output_dir, eval_name)
        if check_path_exists(destination) and not no_skip:
            logger.info(f"Eval exists at: {destination}...skipping")
            continue
        os.chdir(eval_dir)
        model_checkpoint = download_checkpoint(
            model_row,
            eval_dir,
            prefix_replacement,
            tri_s3_path,
            profile=profile,
            checkpoint_replacement=checkpoint_replacement,
        )
        params_file = download_params(model_row, eval_dir, prefix_replacement, tri_s3_path, profile=profile)
        if model_row["hyperparameters.model"].startswith("open_lm"):  # check for open lm config
            model_config = f"{model_row['hyperparameters.model']}"
        else:
            model_config = f"{CWD}/{model_row['hyperparameters.model']}"
        eval_script = f"{CWD}/eval/eval_openlm_ckpt.py"
        if model_checkpoint and params_file:
            eval_output = run_eval(
                eval_script,
                model_checkpoint,
                tokenizer,
                model_config,
                eval_yaml,
                params_file,
                output_dir,
                skip_perplexity,
                averager_name,
                hf_model,
                hf_cache_dir,
                num_gpus,
                force_xformers,
            )
            shutil.rmtree(eval_dir)
            os.makedirs(eval_dir)
            os.chdir(CWD)
            modify_and_save_evaluation_json(eval_output, model_row["uuid"], output_dir, eval_name)


if __name__ == "__main__":
    main()
