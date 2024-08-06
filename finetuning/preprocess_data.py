import jsonlines
import glob
import os
import threading
from webdataset import ShardWriter
import random
import time
import io
import zstandard as zstd
from contextlib import contextmanager
import argparse
from pathlib import Path
from transformers import GPTNeoXTokenizerFast
import numpy as np

QUEUE_MAX = 10000
BUFFER_MIN = 1000
BUFFER_MAX = 200000
CHUNK_SIZE = 2048 + 1
SHARD_SIZE = None
SLEEP_TIME = 1
TARGET_MASK_LEFT = 50300
TARGET_MASK_INDIVIDUAL = 50400

PROMPT = '''Below is an instruction that describes a task.\n\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:'''

eot_token = "<|endoftext|>"

def write_to_shard(chunks, shard_writer):
    for idx, chunk in enumerate(chunks):
        shard_writer.write({"__key__": f"{idx:12d}", "txt": str(chunk)})


@contextmanager
def get_item_reader(file_name):
    if file_name.endswith(".jsonl"):
        with jsonlines.open(file_name) as reader:
            yield reader
    else:
        dctx = zstd.ZstdDecompressor()
        with open(file_name, "rb") as compressed_file:
            with dctx.stream_reader(compressed_file) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
                    with jsonlines.Reader(text_reader) as jsonl_reader:
                        yield jsonl_reader


def process_files(file_list, buffer, enc, buffer_lock):
    remaining_tokens = []
    queue = []

    def dump_queue_to_buffer():
        with buffer_lock:
            while queue:
                buffer.append(queue.pop(0))

    for file_name in file_list:
        print("Processing", file_name)

        with get_item_reader(file_name) as item_reader:
            for item in item_reader:
                instruction = item['instruction']
                instruction_w_prompt = PROMPT.format(instruction = instruction)
                response = item['response']
                try:
                    tokens = enc(instruction_w_prompt) + [TARGET_MASK_LEFT] + enc(response) + [eot_token]   
                    if len(tokens) < CHUNK_SIZE:
                        diff = CHUNK_SIZE - len(tokens)       
                        tokens = tokens + (diff * [TARGET_MASK_INDIVIDUAL]) 
                    else:
                        tokens = tokens[:CHUNK_SIZE]
                except:
                    print("Failed to encode string.")
                    continue
                    
                if len(buffer) > BUFFER_MAX:
                    time.sleep(1)
                    continue

                if buffer_lock.locked():
                    if len(queue) < QUEUE_MAX:
                        queue.append(tokens)
                    else:
                        time.sleep(1)
                else:
                    if queue:
                        dump_queue_to_buffer()
                    with buffer_lock:
                        buffer.append(tokens)


def consumer(my_id, output_dir, threads, buffer, buffer_lock, num_consumers):
    output_directory = f"{output_dir}/{CHUNK_SIZE - 1}-v1/{my_id}"
    os.makedirs(output_directory, exist_ok=True)
    shard_writer = ShardWriter(os.path.join(output_directory, "shard-%07d.tar"), maxcount=SHARD_SIZE)

    chunks = []

    start_time = time.time()

    while any(t.is_alive() for t in threads):
        time.sleep(SLEEP_TIME)
        with buffer_lock:
            lenb = len(buffer)
            print("Length of buffer", lenb)
            if lenb >= BUFFER_MIN:
                while buffer and len(chunks) < SHARD_SIZE:
                    random_index = random.randint(0, len(buffer) - 1)
                    chunks.append(buffer[random_index])
                    buffer.pop(random_index)  # Remove the selected element

        if len(chunks) == SHARD_SIZE:
            print(f"I am {my_id} and I am writing a shard.", len(buffer))
            write_to_shard(chunks, shard_writer)
            # print("FNAME", shard_writer.fname)
            chunks = []
            time_for_shard = time.time() - start_time
            print("shards / s", num_consumers / time_for_shard)
            print("tokens / s", num_consumers * SHARD_SIZE * CHUNK_SIZE / time_for_shard)
            print(
                "hours req for 1.2T tokens",
                1_200_000_000_000 / (num_consumers * SHARD_SIZE * CHUNK_SIZE / time_for_shard) / 3600,
            )

            start_time = time.time()

    # Process the remaining items in the buffer after all threads have completed
    while buffer:
        with buffer_lock:
            while buffer and len(chunks) < SHARD_SIZE:
                random_index = random.randint(0, len(buffer) - 1)
                chunks.append(buffer[random_index])
                buffer.pop(random_index)  # Remove the selected element

        write_to_shard(chunks, shard_writer)
        chunks = []


def tokenize_eleutherai(tokenizer, string):
    return tokenizer(string).input_ids

def count_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def main(
    input_files,
    output_dir,
    num_workers=32,
    num_consumers=8,
):
    os.makedirs(f"{output_dir}/tars-{CHUNK_SIZE - 1}-v1", exist_ok=True)

    input_files = [glob.glob(input_file) for input_file in input_files]
    input_files = [x for y in input_files for x in y]

    # Shuffle the input files
    random.shuffle(input_files)

    print("Input files", input_files)
    total_entries = 0
    for input_file in input_files:
        total_entries += count_lines(input_file)

    global SHARD_SIZE 
    SHARD_SIZE = min(np.ceil(total_entries/8), 8192) 
    enc = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    tokenize = lambda x: tokenize_eleutherai(enc, x)
    buffer = []  # Use list instead of queue.Queue
    buffer_lock = threading.Lock()

    files_per_worker = len(input_files) // num_workers
    threads = []
    for i in range(num_workers):
        start = i * files_per_worker
        end = (i + 1) * files_per_worker if i < num_workers - 1 else len(input_files)
        t = threading.Thread(
            target=process_files,
            args=(input_files[start:end], buffer, tokenize, buffer_lock),
        )
        t.start()
        threads.append(t)

    consumer_threads = []
    for i in range(num_consumers):
        t = threading.Thread(
            target=consumer,
            args=(
                i,
                output_dir,
                threads,
                buffer,
                buffer_lock,
                num_consumers,
            ),
        )
        t.start()
        consumer_threads.append(t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-files", type=str, nargs="+", help="Input directory containing jsonl files for finetuning.")
    parser.add_argument("--output-dir", type=Path, help="Output directory for tokenized data.")
    parser.add_argument("--num-workers", type=int, default=32, help="Number of workers for tokenization.")
    parser.add_argument("--num-consumers", type=int, default=8, help="Number of writer workers.")

    args = parser.parse_args()

    main(
        args.input_files,
        args.output_dir,
        args.num_workers,
        args.num_consumers,
    )
