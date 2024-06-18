"""
S3 Object Migration Tool

This script provides functionality to migrate objects from one AWS S3 bucket to another,
using specified AWS profiles for source and destination buckets. It's designed to handle
the process in three steps: copying the object to a temporary location, moving it from
there to the destination, and then deleting the object from the temporary location.

The script is built to work with a specific table structure, where it reads a specified
column containing S3 keys (URLs) and processes each key accordingly.

Functions:
- object_exists(bucket, key, aws_profile): Checks if an object exists in a given S3 bucket.
- migrate_object(s3_key, source_profile, dest_profile, tmp_bucket, dest_bucket, tmp_prefix, dest_prefix):
    Handles the migration of a single object from source to destination via a temporary location.

Usage:
The script is run from the command line with several options:
--source_profile: AWS profile name for the source bucket.
--dest_profile: AWS profile name for the destination bucket.
--tmp_location: Temporary S3 location (in the format s3://bucket/prefix) used during migration.
--dest_location: Final destination S3 location (in the format s3://bucket/prefix).
--table: The name of the table to process.
--column: The name of the column in the table that contains the S3 keys.
--database_path: Path to the database directory containing the table data.

The AWS CLI must be installed and configured with the necessary profiles for this script to work.
The script uses the AWS CLI for S3 operations and subprocesses for executing these CLI commands.

Example:
python migrate_script.py --source_profile sourceProfile --dest_profile destProfile \
--tmp_location s3://tempbucket/prefix --dest_location s3://destbucket/prefix \
--table mytable --column s3keycolumn --database_path /path/to/database

This command will process 'mytable', looking for S3 keys in 's3keycolumn', and migrate them
from the source to the destination, using 'sourceProfile' and 'destProfile' for AWS operations.

Note:
- Proper AWS credentials and permissions are required for the source, temporary, and destination buckets.
- Ensure that the AWS CLI is installed and configured correctly.
- The script logs each step of the migration process for monitoring and debugging purposes.

Testing in a controlled environment before use in production is strongly recommended to avoid unintended data loss.
"""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import boto3
import click
from expdb import build_table_dfs, filter_df
from loguru import logger
from tqdm import tqdm


def object_exists(bucket, key, aws_profile):
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile
    result = subprocess.run(["aws", "s3", "ls", f"s3://{bucket}/{key}"], capture_output=True, env=env)
    return result.returncode == 0


def migrate_object(s3_key, source_profile, dest_profile, tmp_bucket, dest_bucket, tmp_prefix, dest_prefix, sync):
    parsed_url = urlparse(s3_key)
    source_bucket = parsed_url.netloc
    object_key = parsed_url.path.lstrip("/")

    tmp_key = f"s3://{tmp_bucket}/{tmp_prefix}/{object_key}"
    dest_key = f"s3://{dest_bucket}/{dest_prefix}/{object_key}"

    # Check if object exists in destination
    cmd = "sync" if sync else "cp"
    if not sync and object_exists(dest_bucket, f"{dest_prefix}/{object_key}", dest_profile):
        logger.info(f"Object {dest_key} already exists in destination. Skipping migration.")
        return None, None
    env = os.environ.copy()
    # Copy to temporary location
    env["AWS_PROFILE"] = source_profile
    subprocess.run(["aws", "s3", cmd, s3_key, tmp_key], check=True, env=env)
    logger.info(f"Copied {s3_key} to temporary location {tmp_key}")

    # Copy from temporary to destination
    env["AWS_PROFILE"] = dest_profile
    subprocess.run(["aws", "s3", cmd, tmp_key, dest_key], check=True, env=env)
    logger.info(f"Copied from temporary location {tmp_key} to destination {dest_key}")

    # Delete temporary object
    if sync:
        del_cmd = ["rm", "--recursive"]
    else:
        del_cmd = ["rm"]
    subprocess.run(["aws", "s3"] + del_cmd + [tmp_key], check=True, env=env)
    logger.info(f"Deleted temporary object {tmp_key}")

    return s3_key, dest_key


@click.command()
@click.option("--source_profile", required=True, help="Source AWS profile")
@click.option("--dest_profile", required=True, help="Destination AWS profile")
@click.option("--tmp_location", required=True, help="Temporary S3 location (s3://bucket/prefix)")
@click.option("--dest_location", required=True, help="Destination S3 location (s3://bucket/prefix)")
@click.option("--table", required=True, help="Table to process")
@click.option("--column", required=True, help="Column containing S3 keys")
@click.option("--database_path", default="exp_data", help="Path to the database directory")
@click.option("--filter", "-f", multiple=True, help="Filter rows based on column conditions")
@click.option("--update-json", is_flag=True, help="Update path in json to copied path")
@click.option("--sync", is_flag=True, help="use aws s3 sync instead of cp")
def main(
    source_profile, dest_profile, tmp_location, dest_location, table, column, database_path, filter, update_json, sync
):
    source_session = boto3.Session(profile_name=source_profile)
    dest_session = boto3.Session(profile_name=dest_profile)
    tmp_bucket = tmp_location.replace("s3://", "").split("/", 1)[0]
    dest_bucket = dest_location.replace("s3://", "").split("/", 1)[0]
    tmp_prefix = tmp_location.replace(f"s3://{tmp_bucket}/", "")
    dest_prefix = dest_location.replace(f"s3://{dest_bucket}/", "")
    num_cores = os.cpu_count()

    source_client = source_session.client("s3")
    dest_client = dest_session.client("s3")

    # Load DataFrame using build_table_dfs from expdb
    table_dfs = build_table_dfs(database_path)
    df = table_dfs[table]
    if filter:
        df = filter_df(df, filter)

    if "checkpoint_url" in df.columns:
        df = df.copy()
        df["stats_url"] = df["checkpoint_url"].str.replace("epoch_", "stats_")

    s3_keys = df[column].dropna().unique()

    with ThreadPoolExecutor() as executor, tqdm(total=num_cores) as progress:
        futures = []
        for s3_key_list in s3_keys:
            if not isinstance(s3_key_list, list):
                s3_key_list = [s3_key_list]
            for s3_key in s3_key_list:
                assert isinstance(s3_key, str)
                if s3_key.startswith("s3://"):
                    future = executor.submit(
                        migrate_object,
                        s3_key,
                        source_profile,
                        dest_profile,
                        tmp_bucket,
                        dest_bucket,
                        tmp_prefix,
                        dest_prefix,
                        sync,
                    )
                    futures.append(future)
                else:
                    logger.warning(f"Invalid S3 key: {s3_key}")
        for future in futures:
            src_key, dest_key = future.result()
            if update_json and src_key is not None:
                file_paths = df[df[column] == src_key]["_source_json"]
                for file_path in file_paths:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    if data.get(column, "") != src_key:
                        print(f"Could not find {column} in {file_path}, not updating.")
                        continue
                    data[column] = dest_key
                    with open(file_path, "w") as f:
                        json.dump(data, f, indent=4)
            progress.update(1)


if __name__ == "__main__":
    main()
