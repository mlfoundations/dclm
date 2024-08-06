from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
import os
import json
import jsonlines
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":

    os.makedirs("sft_data")

    test_dataset = load_dataset("nvidia/OpenMathInstruct-1", split="train")
    test_dataset = test_dataset.rename_column("question", "instruction")
    test_dataset = test_dataset.rename_column("expected_answer", "response")
    test_dataset.to_json("sft_data/OpenMathInstruct/OpenMathInstruct.jsonl")

    print("Data saved in sft_data/OpenMathInstruct/OpenMathInstruct.jsonl")