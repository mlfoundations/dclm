from datasets import load_dataset
import io
import gzip
import math
from tqdm import tqdm

import multiprocessing as mp


def helper(args):
    idx, split_dataset = args
    fname = f"/tmp2/fineweb-edu-sample-350BT/fineweb-edu-sample-350BT-{idx:06}.jsonl.gz"
    split_dataset.to_json(fname, compression="gzip")
    return 0


if __name__ == "__main__":
    splits, examples = 50000, 339347842  # for sample-350BT
    # splits, examples = 1000, 9672101 # for sample-10BT

    bs = math.ceil(examples / splits)

    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-350BT",
        split=[f"train[{k*bs}:{(k+1)*bs}]" for k in range(0, splits)],
        streaming=False,
    )
    with mp.Pool(128) as pool:
        data = []
        for worker_data in tqdm(pool.imap_unordered(helper, [(idx, sp) for idx, sp in enumerate(fw)])):
            data.append(worker_data)
