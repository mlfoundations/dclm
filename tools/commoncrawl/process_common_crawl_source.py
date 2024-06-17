import argparse
import gzip
import json
import os
import pathlib
from io import BytesIO

import boto3
from fastwarc.stream_io import FileStream, GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType


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
        with open_file(output_path, "wb") as f:
            f.put(Body=gzip.compress("\n".join(data).encode("utf-8")))
    else:
        with gzip.open(output_path, mode) as f:
            f.write("\n".join(data))


def process_file(path, documents_per_jsonl, is_wet, output_dir):
    output_file_template = os.path.join(output_dir, os.path.basename(path).replace(".gz", "") + "_{}.jsonl.gz")

    with GZipStream(open_file(path)) as stream:
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
                print(latest_warcinfo)
                continue

            record_data = {
                "text": record.reader.read().decode("utf-8").strip(),
                "metadata": dict(record.headers),
            }

            if latest_warcinfo:
                record_data["warcinfo"] = latest_warcinfo

            jsonl_content.append(json.dumps(record_data))
            count += 1

            if count >= documents_per_jsonl:
                output_path = output_file_template.format(file_index)
                write_output(output_path, jsonl_content)
                file_index += 1
                count = 0
                jsonl_content = []

        if jsonl_content:
            output_path = output_file_template.format(file_index)
            write_output(output_path, jsonl_content)


def load_json_file(json_file_path):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    return data.get("dataset_urls", [])


def main():
    parser = argparse.ArgumentParser(description="Convert WARC/WET to JSONL.GZ with metadata")
    parser.add_argument("json_file_path", help="Path to the JSON file containing WARC/WET paths")
    parser.add_argument(
        "--documents_per_jsonl",
        type=int,
        default=1000,
        help="Number of documents per JSONL file",
    )
    parser.add_argument("--wet", action="store_true", help="Indicate if the files are WET format")
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory (local or s3://bucket/path)",
    )
    parser.add_argument("--subset", type=int, default=None, help="Process only a subset of file paths")
    args = parser.parse_args()
    # Create output directory if it doesn't exist
    if not args.output_dir.startswith("s3://"):
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    file_paths = load_json_file(args.json_file_path)
    if args.subset:
        file_paths = file_paths[: args.subset]

    for file_path in file_paths:
        if args.wet:
            file_path = convert_warc_to_wet(file_path)
        process_file(file_path, args.documents_per_jsonl, args.wet, args.output_dir)


if __name__ == "__main__":
    main()
