import gzip
import io
import json

import boto3
from tqdm import tqdm

# computed using wet file tokenization of CC-MAIN-2019-18 dump (C4 source)
APPROX_TOKENS_PER_WARC = 90000000


def get_commoncrawl_paths():
    s3_client = boto3.client("s3")
    bucket_name = "commoncrawl"
    base_prefix = "crawl-data/"
    paths = []
    # Create a paginator for listing objects
    paginator = s3_client.get_paginator("list_objects_v2")
    # Use the paginator to list all the crawl directories
    for page in paginator.paginate(Bucket=bucket_name, Prefix=base_prefix, Delimiter="/"):
        for crawl in tqdm(page.get("CommonPrefixes", [])):
            crawl_prefix = crawl["Prefix"]
            warc_paths_file = crawl_prefix + "warc.paths.gz"

            # Download and uncompress the warc.paths.gz file
            try:
                warc_paths_object = s3_client.get_object(Bucket=bucket_name, Key=warc_paths_file)
                with gzip.GzipFile(fileobj=io.BytesIO(warc_paths_object["Body"].read())) as gz:
                    # Decode and split the content into paths
                    crawl_paths = [f"s3://{bucket_name}/{x}" for x in gz.read().decode("utf-8").splitlines()]
                    paths.extend(crawl_paths)
            except s3_client.exceptions.NoSuchKey:
                print(f"File not found: {warc_paths_file}")
                continue

    return paths


if __name__ == "__main__":
    # Example usage
    paths = get_commoncrawl_paths()
    with gzip.open("commoncrawl_paths.txt.gz", "wt") as gz_file:
        for path in paths:
            gz_file.write(path + "\n")
