import argparse
import boto3
import os
import multiprocessing
from queue import Queue
import queue
from tqdm import tqdm
import threading
from huggingface_hub import HfApi, HfFolder, CommitOperationAdd, preupload_lfs_files, create_commit
from loguru import logger
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Sync S3 files to Hugging Face repository")
    parser.add_argument("--s3_bucket", type=str, default="***REMOVED***", help="S3 bucket name")
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default="users/vaishaal/mlr/hero-run-fasttext/filtered/OH_eli5_vs_rw_v2_bigram_200k_train/fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train/processed_data/",
        help="S3 prefix",
    )
    parser.add_argument(
        "--hf_repo_id", type=str, default="mlfoundations/dclm-baseline-4T", help="Hugging Face repository ID"
    )
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face access token")
    parser.add_argument(
        "--local_dir", type=str, default="s3_files", help="Local directory to store S3 files temporarily"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=multiprocessing.cpu_count() // 4,
        help="Number of processes for downloading and uploading",
    )
    parser.add_argument("--batch_size", type=int, default=100, help="Number of files to upload in each batch")
    return parser.parse_args()


def list_s3_objects(s3, bucket, prefix):
    logger.info(f"Listing objects in S3 bucket: {bucket} with prefix: {prefix}")
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" in page:
            objects.extend(page["Contents"])
    return objects


def list_hf_files(api, repo_id):
    logger.info(f"Listing files in Hugging Face repository: {repo_id}")
    hf_files = api.list_repo_files(repo_id, repo_type="dataset")
    return set(hf_files)


def download_worker(s3, bucket, local_dir, download_queue, upload_queue):
    while not download_queue.empty():
        s3_key = download_queue.get()
        local_file_path = os.path.join(local_dir, os.path.basename(s3_key))
        logger.info(f"Downloading {s3_key} to {local_file_path}")
        s3.download_file(bucket, s3_key, local_file_path)
        upload_queue.put((local_file_path, s3_key))
        download_queue.task_done()


def upload_worker(api, repo_id, s3_prefix, upload_queue, batch_size):
    batch = []
    local_file_paths = []
    while True:
        try:
            local_file_path, s3_key = upload_queue.get(timeout=300)
            logger.info(f"{local_file_path}/{s3_key}")
            hf_repo_path = s3_key[len(s3_prefix) :]
            logger.info(f"Adding {local_file_path} to Hugging Face at commit {hf_repo_path}")
            batch.append(CommitOperationAdd(path_in_repo=hf_repo_path, path_or_fileobj=local_file_path))
            local_file_paths.append(local_file_path)

            print(len(batch), batch_size)
            if len(batch) >= batch_size:
                logger.info(f"uploading batch: {batch}")
                preupload_lfs_files(repo_id, additions=batch, repo_type="dataset")
                create_commit(
                    repo_id, operations=batch, commit_message=f"Batch commit of {len(batch)} files", repo_type="dataset"
                )
                for fp in local_file_paths:
                    logger.info(f"removing: {fp}")
                    os.remove(fp)
                batch.clear()
                local_file_paths.clear()
            upload_queue.task_done()
        except queue.Empty:
            print("potato queue empty!")
            if batch:
                preupload_lfs_files(repo_id, additions=batch, repo_type="dataset")
                create_commit(
                    repo_id, operations=batch, commit_message=f"Batch commit of {len(batch)} files", repo_type="dataset"
                )
                for addition in batch:
                    if addition.path_or_fileobj:
                        os.remove(addition.path_or_fileobj)
                for fp in local_file_paths:
                    logger.info(f"removing: {fp}")
                    os.remove(fp)
                batch.clear()
                local_file_paths.clear()
                raise
        except Exception as e:
            print("potato")
            logger.error(f"Error uploading file: {e}")
            raise


def main():
    args = parse_args()

    # Initialize S3 client
    s3 = boto3.client("s3")

    # Initialize Hugging Face API
    api = HfApi()
    hf_folder = HfFolder()
    hf_folder.save_token(args.hf_token)

    # Create a local directory to store S3 files temporarily
    os.makedirs(args.local_dir, exist_ok=True)

    # List objects in the specified S3 path
    objects = list_s3_objects(s3, args.s3_bucket, args.s3_prefix)
    logger.info(f"total number of objects: {len(objects)}")

    # List files in the Hugging Face repository
    hf_file_set = list_hf_files(api, args.hf_repo_id)
    logger.info(f"total number of files in hugging face: {len(hf_file_set)}")

    # Create queues
    download_queue = Queue()
    upload_queue = Queue()
    total_to_download = 0
    # Fill the download queue with S3 keys, skipping existing files in Hugging Face
    for obj in objects:
        s3_key = obj["Key"]
        hf_repo_path = s3_key[len(args.s3_prefix) :]
        if hf_repo_path not in hf_file_set:
            download_queue.put(s3_key)
            total_to_download += 1

    logger.info(f"Total to download: {total_to_download}")
    # Determine the number of CPU cores
    num_processes = args.num_processes

    # Create and start download processes
    download_processes = []
    for _ in range(num_processes):
        p = threading.Thread(
            target=download_worker, args=(s3, args.s3_bucket, args.local_dir, download_queue, upload_queue)
        )
        p.start()
        download_processes.append(p)

    # Create and start upload processes
    upload_processes = []
    for _ in range(num_processes):
        p = threading.Thread(
            target=upload_worker, args=(api, args.hf_repo_id, args.s3_prefix, upload_queue, args.batch_size)
        )
        p.start()
        upload_processes.append(p)

    # Wait for all tasks to complete
    download_queue.join()
    upload_queue.join()

    logger.info("Finished sleeping to let processes finish...")
    time.sleep(300)
    # Terminate processes
    for p in download_processes + upload_processes:
        p.terminate()


if __name__ == "__main__":
    main()
