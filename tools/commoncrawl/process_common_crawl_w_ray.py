import argparse
import gzip
import hashlib
import json
import os
import time
import traceback
from typing import BinaryIO, List
import pathlib
from io import BytesIO
from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType


import boto3
import botocore
import pandas as pd
import ray
from ray._private.internal_api import memory_summary
from ray.data.context import DataContext
import random
import trafilatura
from resiliparse.extract.html2text import extract_plain_text


def convert_warc_to_wet(warc_path): 
    return warc_path.replace("/warc/", "/wet/").replace(".warc.gz", ".warc.wet.gz")


def open_file(path, mode="rb"):
    if path.startswith("s3://"):
        s3 = boto3.resource("s3")
        bucket_name, key = path[5:].split("/", 1)
        obj = s3.Object(bucket_name, key)
        if mode == "rb":
            return BytesIO(obj.get()["Body"].read())
        else:
            return obj
    else:
        return FileStream(path, mode)


def write_output(output_path, data, mode="wt"):
    if output_path.startswith("s3://"):
        f = open_file(output_path, "wb")
        f.put(Body=gzip.compress("\n".join(data).encode("utf-8")))
    else:
        with gzip.open(output_path, mode) as f:
            f.write("\n".join(data))


@ray.remote
class GlobalCounter:
    def __init__(self):
        self.value = 0
        self.token_count = 0

    def increment(self):
        self.value += 1
        return self.value

    def increment_token_count(self, num_tokens):
        self.token_count += num_tokens
        return self.token_count

    def get_counter(self):
        return self.value

    def get_token_counter(self):
        return self.token_count

def process_file_batch(path, documents_per_jsonl, is_wet, output_dir, counter):
    jitter = random.uniform(1, 30)
    time.sleep(jitter)
    s3 = boto3.resource('s3')
    rets = []
    for p in path["path"]:
        ret = process_file(s3, p, documents_per_jsonl, is_wet, output_dir, counter)
        rets.append(ret)
    return {"data": rets}


def process_file(s3, path, documents_per_jsonl, is_wet, output_dir, counter):


    if args.wet:
        path = convert_warc_to_wet(path)
    s = time.time()

    # basename alone has collisions
    short_md5 = hashlib.md5(path.encode()).hexdigest()[:7]
    assert path.endswith(".gz")
    hash_path = path[:-3] + f"_{short_md5}" + ".gz"

    output_file_check = os.path.join(
        output_dir.rstrip('/') + "_check", os.path.basename(hash_path).replace(".gz", "") + ".stat"
    )
    s3 = boto3.resource('s3')

    try:
        check_bucket, check_key = output_file_check[5:].split("/", 1)
        check_obj = s3.Object(check_bucket, check_key).load()
        wet_count = ray.get(counter.increment_token_count.remote(1))

        if wet_count % 1000 == 0: 
            print(f"Seen wet count {wet_count}")
        return [{"time": 0}]
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            # The object does not exist.
            pass
        else:
            # Something else has gone wrong.
            # For now, recompute
            pass

    if documents_per_jsonl is not None:
        output_file_template = os.path.join(
            output_dir, os.path.basename(hash_path).replace(".gz", "") + "_{}.jsonl.gz"
        )
    else:
        output_file_template = os.path.join(
            output_dir, os.path.basename(hash_path).replace(".gz", "") + ".jsonl.gz"
        )

    num_tries = 0
    delay = 1
    MAX_NUM_TRIES = 10

    while num_tries < MAX_NUM_TRIES:
        try:
            gz_file = open_file(path)
            break
        except:
            num_tries += 1
            backoff = delay * 2**num_tries
            jitter = backoff * random.uniform(0.5, 1.5)
            time.sleep(jitter)

    if num_tries >= MAX_NUM_TRIES:
        print(f"Not found in time: {time.time() - s}")
        return [{"time": time.time() - s}]
    else:
        with GZipStream(gz_file) as stream:
            record_type_filter = (
                WarcRecordType.conversion if is_wet else WarcRecordType.response
            ) | WarcRecordType.warcinfo
            iterator = ArchiveIterator(stream, record_types=record_type_filter)
            count = 0
            file_index = 1
            jsonl_content = []
            latest_warcinfo = None
            for record in iterator:
                if record.record_type == WarcRecordType.warcinfo:
                    latest_warcinfo = record.reader.read().decode("utf-8").strip()
                    # print(latest_warcinfo)
                    continue
                try:
                    record_data = {
                        # "text": record.reader.read().decode("utf-8").strip(),
                        # "text": trafilatura.extract(record.reader.read()), # .decode("utf-8").strip(),
                        "text": extract_plain_text(record.reader.read().decode("utf-8"), main_content=True),
                        "metadata": dict(record.headers),
                    }
                except:
                    continue
                if latest_warcinfo:
                    record_data["warcinfo"] = latest_warcinfo
                jsonl_content.append(json.dumps(record_data))
                count += 1
                if documents_per_jsonl is not None and count >= documents_per_jsonl:
                    output_path = output_file_template.format(file_index)
                    write_output(output_path, jsonl_content)
                    file_index += 1
                    count = 0
                    jsonl_content = []
            if jsonl_content:
                output_path = output_file_template.format(file_index)
                write_output(output_path, jsonl_content)

        wet_count = ray.get(counter.increment_token_count.remote(1))

        if wet_count % 100 == 0:
            print(f"Current wet count {wet_count}")

        print(f"Extracted and converted in time {time.time() - s}")
        write_output(output_file_check, ['done'])

        return [{"time": time.time() - s}]


def load_json_file(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data.get("dataset_urls", [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert WARC/WET to JSONL.GZ with metadata")
    parser.add_argument(
        "--json_file_path", type=str, required=True, help="Path to the JSON file containing WARC/WET paths"
    )
    parser.add_argument(
        "--output_path",
        help="output path",
        type=str,
        required=True,
        ## OLD 
        # e.g s3://dcnlp-west/common_crawl_1e12_approx_tokens_sample_v2_data/
        ## NEW
        # e.g s3://dcnlp-west/common_crawl_v3_pre2023_0.01_frac_sample_jsonls/
    )
    parser.add_argument(
        "--documents_per_jsonl",
        type=int,
        default=None,
        help="Number of documents per JSONL file",
    )
    parser.add_argument("--wet", action="store_true", help="Indicate if the files are WET format")
    parser.add_argument("--subset", type=int, default=None, help="Process only a subset of file paths")
    parser.add_argument("--subset_frac", type=float, default=None, help="Process only a subset fraction of file paths")
    parser.add_argument("--allow_errors", type=int, default=100, help="Ignore errors on these many number of files")
    parser.add_argument("--ray_address", type=str, default=None)
    parser.add_argument("--force_parallelism", type=int, default=None)
    parser.add_argument("--ray_spill_location", type=str, default="/tmp/ray_spill")
    parser.add_argument("--batch_size", default=256, type=int)

    args = parser.parse_args()

    """
    Example usage: 

    python process_common_crawl_w_ray.py --json_file_path CC_200e12_approx_tokens_sample_v3_pre2023.json \
    --output_path s3://dcnlp-west/common_crawl_v3_pre2023_0.01_frac_sample_jsonls/ --documents_per_jsonl 5000 \
    --subset_frac 0.01  --force_parallelism 128 --wet

    python process_common_crawl_w_ray.py --json_file_path CC_200e12_approx_tokens_sample_v3_pre2023.json \
    --output_path s3://dcnlp-west/common_crawl_v4_pre2023_0.15_frac_sample_jsonls/ --documents_per_jsonl 40000 \
    --subset_frac 0.15
    """


    # configure remote spilling
    creds = {k: v for k, v in os.environ.items() if k.startswith("AWS")}
    runtime_env = {"env_vars": creds}
    # runtime_env = {}

    # if args.ray_address is None:
    #     ray.init(runtime_env=runtime_env, _temp_dir=args.ray_spill_location)
    # else:
    #     ray.init(address=args.ray_address, runtime_env=runtime_env, _temp_dir=args.ray_spill_location)
    
    ray.init(address="auto", runtime_env=runtime_env,)

    num_nodes = len(ray.nodes())

    # Create output directory if it doesn't exist
    if not args.output_path.startswith("s3://"):
        pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    file_paths = load_json_file(args.json_file_path)

    if args.subset:
        file_paths = file_paths[: args.subset]

    if not args.subset and args.subset_frac:
        file_paths = file_paths[: int(args.subset_frac * len(file_paths))]

    print(f"num paths ={len(file_paths)}")

    num_files = len(file_paths)
    num_cores = os.cpu_count()

    output_path = args.output_path

    if args.force_parallelism is not None:
        parallelism = args.force_parallelism
    else:
        parallelism = num_cores * num_nodes

    ctx = DataContext.get_current()
    ctx.execution_options.resource_limits.object_store_memory = float("inf")
    ctx.max_errored_blocks = args.allow_errors
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    counter = GlobalCounter.remote()

    start_time = time.time()

    ds = ray.data.from_pandas(pd.DataFrame(file_paths, columns=["path"])).repartition(parallelism)
    print("ds count=", ds.count())

    ds = ds.map_batches(
        lambda x: process_file_batch(
            x,
            documents_per_jsonl=args.documents_per_jsonl,
            is_wet=args.wet,
            output_dir=args.output_path,
            counter=counter
        ), batch_size=args.batch_size
    ).count()
    end_time = time.time()

    final_wet_count = ray.get(counter.increment_token_count.remote(0))

    duration = end_time - start_time
    print(f"Script finished in: {duration}")
    print(f"Final wet count: {final_wet_count}")
    try:
        print(memory_summary(stats_only=True))
    except Exception:
        print("Failed to retrieve memory summary")
        print(traceback.format_exc())

    print("")
