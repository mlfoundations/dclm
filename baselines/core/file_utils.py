import json
from typing import List
import zstandard as zstd
import gzip
import jsonlines
import io
from typing import BinaryIO, List

from cloudpathlib import S3Path
import os
import boto3
from pathlib import Path as LocalPath


def is_s3(file_path: str):
    return file_path.startswith("s3://")

def is_compressed(file_path: str):
    return any(file_path.endswith(z) for z in (".zst", ".zstd", ".gz"))

def delete_file(file_path: str):
    """Deletes the file at the given path (local or S3). If the file does not exist, raises an error.
    May also raise if this is a directory rather than a file"""
    if is_s3(file_path):
        s3_path = S3Path(file_path)
        if s3_path.exists():
            s3_path.unlink()  # This deletes the file
        else:
            raise FileNotFoundError(f"{file_path} does not exist.")
    else:
        os.remove(file_path)


def is_exists(file_path: str):
    """Checks if the file at the given path (local or S3) exists"""
    if is_s3(file_path):
        s3_path = S3Path(file_path)
        return s3_path.exists() and s3_path.is_file()
    else:
        return os.path.isfile(file_path)


def _jsonl_bytes_reader(fh: BinaryIO):
    with io.TextIOWrapper(fh, encoding="utf-8") as text_reader:
        with jsonlines.Reader(text_reader) as jsonl_reader:
            for item in jsonl_reader:
                yield item

def read_jsonl(file_path: str):
    """Read a JSONL file from a given path (local or S3)."""
    if is_s3(file_path):
        path = S3Path(file_path)
    else:
        path = LocalPath(file_path)

    if any(file_path.endswith(z) for z in (".zst", ".zstd")):
        with path.open('rb') as f:
            with zstd.ZstdDecompressor().stream_reader(f) as reader:
                for line in _jsonl_bytes_reader(reader):
                    yield line
    elif file_path.endswith(".gz"):
        with gzip.open(path, 'rb') as f:
            for line in _jsonl_bytes_reader(f):
                yield line
    else:
        with path.open('rb') as f:    
            for line in _jsonl_bytes_reader(f):
                yield line

def write_jsonl(data, file_path: str, mode: str = "w"):
    """Write data to a JSONL file at a given path (local or S3)."""
    if is_s3(file_path):
        path = S3Path(file_path)
    else:
        path = LocalPath(file_path)

    if is_compressed(file_path):
        data = [json.dumps(d) for d in data]
        data = "\n".join(data).encode('utf8')

    if any(file_path.endswith(z) for z in (".zst", ".zstd")):
        with path.open("wb") as f:
            with zstd.ZstdCompressor().stream_writer(f) as writer:
                writer.write(data)
    elif file_path.endswith(".gz"):
        with path.open("wb") as f:
            f.write(gzip.compress(data))
    else:
        with path.open(mode) as f:
            for item in data:
                json_str = json.dumps(item)
                f.write(f"{json_str}\n")

def makedirs_if_missing(dir_path: str):
    """
    Create directories for the provided path if they do not exist.

    For S3 paths, this function is a no-op because S3 does not have a real notion of directories.

    Parameters:
    - dir_path (str): The directory path. Can be a local filesystem path or an S3 URI.

    Returns:
    - None
    """
    if is_s3(dir_path):
        return  # In S3, directories are virtual and created on-the-fly.
    os.makedirs(dir_path, exist_ok=True)


# util functions for general use
def list_dir(dirname) -> List[str]:
    """List the contents of a directory, excluding hidden files and directories. always as full abs path,
    alawys as list of strings
    """
    if is_s3(dirname):
        s3_directory = S3Path(dirname)
        return [str(f) for f in s3_directory.iterdir() if not f.name.startswith(".")]  # exclude hidden files
    else:
        return [os.path.abspath(os.path.join(dirname, f)) for f in os.listdir(dirname) if not f.startswith(".")]
