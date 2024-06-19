import os
import resource
import time
from typing import List
import traceback
import subprocess

import boto3
import numpy as np
import psutil
from cloudpathlib import S3Path

import ray
from ray._private.internal_api import memory_summary
from ray.data._internal.util import _check_pyarrow_version
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.data.datasource import Datasource, ReadTask

from baselines.core.file_utils import write_jsonl, read_jsonl, makedirs_if_missing

from collections import defaultdict
import hashlib
from io import BytesIO
import tarfile
import simdjson
import zstandard as zstd
import gzip

from baselines.mappers.core_utils import DEDUP_NORMALIZERS

import argparse

# Helpers to deal with cases where content_key emits only one value (e.g. for url dedup)
split_helper = lambda ck: ck if isinstance(ck, list) else [ck]
join_helper = lambda ck: ck[0]


def tar_to_entries(
    batch, input_overlap, content_key="text", normalize=None, selection_key=None, selection_normalize=None
):
    all_rows = {"uid": [], "s3_filename": [], "local_index": []}
    if selection_key is not None:
        all_rows[selection_key] = []

    for idx in range(len(batch["bytes"])):
        if batch["path"][idx].endswith(".zstd"):
            with zstd.ZstdDecompressor().stream_reader(batch["bytes"][idx]) as reader:
                batch["bytes"][idx] = reader.read()

        jsonl_bytes = batch["bytes"][idx]
        jsons = [simdjson.loads(j) for j in jsonl_bytes.decode().splitlines()]
        if normalize is not None:
            uids = [
                hashlib.md5(str(normalize(content_unit)).encode()).hexdigest()
                for j in jsons
                for content_unit in split_helper(j[content_key])
            ]
        else:
            uids = [
                hashlib.md5(content_unit.encode("utf-8")).hexdigest()
                for j in jsons
                for content_unit in split_helper(j[content_key])
            ]
        path = batch["path"][idx]
        path = path[len(input_overlap) :]
        all_rows["uid"].extend(uids)
        all_rows["s3_filename"].extend(len(uids) * [path])
        # could combine below with uids generation and break into two lists after
        all_rows["local_index"].extend(
            [
                (file_index, content_index)
                for file_index, j in enumerate(jsons)
                for content_index in range(len(split_helper(j[content_key])))
            ]
        )
        if selection_key is not None:
            selection = [j[selection_key] for j in jsons for content_index in range(len(split_helper(j[content_key])))]
            if selection_normalize is not None:
                selection = [selection_normalize(s) for s in selection]
            all_rows[selection_key].extend(selection)

    return all_rows


def get_dupe_rows(g, selection_key=None, reverse=False):
    # assumption that number of duplicates in group are small and hence can call min/max
    # return list of duplicates outside of keep_idx for future removal
    if len(g) == 1:
        return {}
    if selection_key is None:
        keep_idx = 0
    else:
        keep_fn = np.argmax if reverse else np.argmin
        keep_idx = keep_fn(g[selection_key])
    return {
        # "s3_filename": g["s3_filename"].drop([keep_idx]),
        # "local_index": g["local_index"].drop([keep_idx]),
        "s3_filename": np.concatenate((g["s3_filename"][:keep_idx], g["s3_filename"][keep_idx + 1 :])),
        "local_index": np.concatenate((g["local_index"][:keep_idx], g["local_index"][keep_idx + 1 :])),
    }


def drop_dupe_rows(g, output_path, content_key, input_overlap, local_stats_dir):
    # Download s3_filename from S3.
    # Remove duplicates in local_index.
    # Upload back to s3.
    # (make sure not to upload directly back to the same filename)
    drop_indices = g["local_index"]
    s3_filename = input_overlap + g["s3_filename"][0]
    input_parts = s3_filename.replace("s3://", "").split("/")
    bucket = input_parts.pop(0)
    key = "/".join(input_parts)
    s3 = boto3.resource("s3")
    obj = s3.Object(bucket, key)

    jsonl_bytes_in = obj.get()["Body"].read()
    if any(s3_filename.endswith(z) for z in (".zst", ".zstd")):
        with zstd.ZstdDecompressor().stream_reader(jsonl_bytes_in) as reader:
            jsonl_bytes_in = reader.read()
    elif s3_filename.endswith(".gz"):
        jsonl_bytes_in = gzip.decompress(jsonl_bytes_in)

    jsons_in = [simdjson.loads(j) for j in jsonl_bytes_in.decode().splitlines()]
    num_jsons_in = len(jsons_in)

    # build mapping of drop_indices from file_index to list of content_index
    index_map = defaultdict(list)
    tmp = [index_map[file_index].append(content_index) for file_index, content_index in drop_indices]

    for file_index in range(len(jsons_in)):
        jsons_in[file_index][content_key] = split_helper(jsons_in[file_index][content_key])

    # remove content
    for file_index, drop_content_indices in index_map.items():
        jsons_in[file_index][content_key] = [
            c
            for content_index, c in enumerate(jsons_in[file_index][content_key])
            if content_index not in drop_content_indices
        ]

    # remove if any json is now empty
    jsons_in = [j for j in jsons_in if len(j[content_key]) != 0]

    # if all rows have single entries, collapse content_key into a string instead of a list, assumes that downstream joiners in local chunks will be able to robustly handle cases where content_key is already not in a list
    all_single_element = all(len(json[content_key]) == 1 for json in jsons_in)
    for file_index in range(len(jsons_in)):
        jsons_in[file_index][content_key] = (
            join_helper(jsons_in[file_index][content_key]) if all_single_element else jsons_in[file_index][content_key]
        )

    json_strs_out = [simdjson.dumps(j) for j in jsons_in]
    # kept = len(json_strs_out) # count jsons kept, not content units
    kept = sum(
        [len(j[content_key]) for j in jsons_in]
    )  # count content units kept, not jsons; len(g["local_index"]) is number dropped
    if any(s3_filename.endswith(z) for z in (".zst", ".zstd")):
        with zstd.ZstdCompressor().stream_reader(("\n".join(json_strs_out)).encode("UTF-8")) as reader:
            jsonl_bytes_out = BytesIO(reader.read())
    elif s3_filename.endswith(".gz"):
        jsonl_bytes_out = BytesIO(gzip.compress(("\n".join(json_strs_out)).encode("UTF-8")))
    else:
        jsonl_bytes_out = BytesIO(("\n".join(json_strs_out)).encode("UTF-8"))

    s3_client = boto3.client("s3")
    output_parts = output_path.replace("s3://", "").split("/")
    out_bucket = output_parts.pop(0)
    output_parts += input_parts[-(len(input_parts) - len(output_parts)) :]
    out_key = "/".join(output_parts)
    s3_client.upload_fileobj(jsonl_bytes_out, out_bucket, out_key)

    # Update local stats files
    shard_name = g["s3_filename"][0].replace("_processed.jsonl", ".jsonl").split(".jsonl")[0]
    stats_out_path = os.path.join(local_stats_dir, shard_name.lstrip("/") + "_stats.jsonl")
    write_jsonl(
        [
            {
                "name": "exact_dedup",
                "content_key": content_key,
                "pages_in": num_jsons_in,
                "pages_out": len(json_strs_out),
            }
        ],
        stats_out_path,
        "a",
    )

    return {"s3_filename": [f"s3://{s3_filename}"], "kept": [kept]}


@ray.remote(max_calls=10)
def write_unmodified_local_stats(s3_filepath, local_stats_dir, input_overlap, content_key):
    s3_filepath = s3_filepath.replace("s3://", "").replace(input_overlap, "")
    shard_name = s3_filepath.replace("_processed.jsonl", ".jsonl").split(".jsonl")[0]
    stats_out_path = os.path.join(local_stats_dir, shard_name.lstrip("/") + "_stats.jsonl")
    write_jsonl(
        [{"name": "exact_dedup", "content_key": content_key, "pages_in": "no_op", "pages_out": "no_op"}],
        stats_out_path,
        "a",
    )


def dedup_jsonl(
    input_dir,
    shard_files=None,
    base_output_path=None,
    working_dir=None,
    sync_to_input=False,
    content_key="text",
    normalize=None,
    selection_key=None,
    selection_normalize=None,
    selection_reverse=False,
):

    ray.init(ignore_reinit_error=True)

    input_overlap = input_dir.replace("s3://", "")
    input_dir_strip = input_dir.rstrip("/")

    # base_output_path is the FINAL output directory for an overall pipeline involving local chunks and dedup
    # Here, it is solely used to locate per-jsonl stats files, while working_dir is where dedup actually outputs to
    base_output_path = input_dir if base_output_path is None else base_output_path
    local_stats_dir = os.path.join(base_output_path, "stats")

    if working_dir is None or input_dir == working_dir:
        working_dir = input_dir_strip + "_working"
    else:
        working_dir.replace("s3://", "").rstrip("/")

    if shard_files is None:
        input_paths = []
        s3_client = boto3.client("s3")
        paginator = s3_client.get_paginator("list_objects_v2")
        input_parts = input_dir_strip.replace("s3://", "").split("/")
        bucket = input_parts.pop(0)
        key = "/".join(input_parts) + "/"
        pages = paginator.paginate(Bucket=bucket, Prefix=key)
        for page in pages:
            try:
                for obj in page["Contents"]:
                    path_body = obj["Key"]
                    if os.path.splitext(path_body)[1] in {".jsonl", ".zstd", ".zst", ".gz"}:
                        input_paths.append(f"s3://{bucket}/{path_body}")
            except KeyError:
                print("No files exist")
                exit(1)
    else:
        input_paths = [f"s3://{input_dir_strip.replace('s3://','')}/{b}" for b in shard_files]

    input_paths = [p for p in input_paths if all(s not in p for s in ["/stats/", "global_stats.jsonl"])]

    ctx = DataContext.get_current()
    ctx.execution_options.resource_limits.object_store_memory = float("inf")
    ctx.use_push_based_shuffle = True
    ray.data.DataContext.get_current().execution_options.verbose_progress = True

    start_time = time.time()

    if normalize is not None and normalize in DEDUP_NORMALIZERS:
        normalize = DEDUP_NORMALIZERS[normalize]
    else:
        normalize = None

    if selection_normalize is not None and selection_normalize in DEDUP_NORMALIZERS:
        selection_normalize = DEDUP_NORMALIZERS[selection_normalize]
    else:
        selection_normalize = None

    tar_to_entries_dict = {
        "input_overlap": input_overlap,
        "content_key": content_key,
        "normalize": normalize,
        "selection_key": selection_key,
        "selection_normalize": selection_normalize,
    }

    ds = ray.data.read_binary_files(input_paths, include_paths=True).map_batches(
        tar_to_entries, batch_size=1, fn_kwargs=tar_to_entries_dict
    )
    exc = None
    ds_stats = None
    try:
        get_dupe_rows_point = lambda g: get_dupe_rows(g, selection_key, selection_reverse)
        ds = ds.groupby("uid").map_groups(get_dupe_rows_point, batch_format="numpy").materialize()
        drop_dupe_rows_point = lambda g: drop_dupe_rows(g, working_dir, content_key, input_overlap, local_stats_dir)
        ds = ds.groupby("s3_filename").map_groups(drop_dupe_rows_point, batch_format="numpy")  # Second sort
        # what if a file has no duplicates? then may not be processed above, so we do a sync below

        ds_final = ds.materialize()
        # kept = ds_final.sum("kept")
        files_written = ds_final.count()
        # print("Kept: " + str(kept))
        ds_stats = ds_final.stats()

    except Exception as e:
        exc = e
        pass

    end_time = time.time()

    duration = end_time - start_time
    print("Finished in", duration)

    if ds_stats is not None:
        print(ds_stats)

    if exc:
        raise exc

    # TODO set expiration policy for working directories
    if files_written == len(input_paths) and not sync_to_input:
        # all files changed, no sync/copy needed
        return working_dir
    else:
        # careful: below sync assumes input_dir is not the original
        sync_list = ["aws", "s3", "sync", working_dir, input_dir]
        process = subprocess.Popen(sync_list)
        process.wait()

        # Write local stats for the unmodified files
        modified_paths = [s["s3_filename"] for s in ds_final.iter_rows()]
        unmodified_paths = [s for s in input_paths if s not in modified_paths]
        ret = [
            write_unmodified_local_stats.remote(u, local_stats_dir, input_overlap, content_key)
            for u in unmodified_paths
        ]
        ray.get(ret)

        return input_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        help="input path",
        type=str,
        required=True,
        # e.g. assumes this is a folder and will find all jsonls within it; dedup will modify this folder
    )
    parser.add_argument(
        "--content_key", type=str, default="text"
    )  # should be a list containing units that should be deduplicated
    parser.add_argument("--selection_key", type=str, default=None)
    parser.add_argument("--normalize", type=str, default=None)

    # normalize and selection_normalize are not supported here because they are functions; use by calling function instead of using cli
    args = parser.parse_args()

    input_dir = args.input
    content_key = args.content_key
    selection_key = args.selection_key

    dedup_jsonl(input_dir, content_key=content_key, selection_key=selection_key, normalize=args.normalize)
