import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import sys

import pandas as pd
import matplotlib.ticker as mtick

# python tools/plot_decontamination.py --json-file-1 exp_data/evals/evaluation_rw_v2_cc_v3_f0.15_resiliparse_fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_0.1-open_lm_7b_swiglutorch-warm=5000-lr=0p001-wd=0p1-cd=3e-05-bs=2048-mult=2-seed=124-tokens=275576422400_heavy.json --json-file-2 exp_data/evals/evaluation_llama2_7b_openlm_heavy.json --contamination-json all_evals_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_0.1.jsonl --contamination-summary all_evals_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_0.1_summary.jsonl --output-file decontamination_plot_llama2.pdf
# python plot_decontamination.py --json-file-1 exp_data/evals/evaluation_rw_v2_cc_v3_f0.15_resiliparse_fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_0.1-open_lm_7b_swiglutorch-warm=5000-lr=0p001-wd=0p1-cd=3e-05-bs=2048-mult=2-seed=124-tokens=275576422400_heavy.json --json-file-2 exp_data/evals/evaluation_llama2_7b_openlm_heavy.json --contamination-json all_evals_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_0.1_summary.jsonl --output-file decontamination_plot_llama2.pdf
# python plot_decontamination.py --json-file-1 exp_data/evals/evaluation_rw_v2_cc_v3_f0.15_resiliparse_fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_0.1-open_lm_1b_swiglutorch-warm=5000-lr=0p003-wd=0p0033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_heavy.json --json-file-2 exp_data/evals/evaluation_rw_v2_w_substr_resiliparse-open_lm_1b-warm=5000-lr=0p003-wd=0p033-cd=3e-05-bs=256-mult=1-seed=124-tokens=28795904000_heavy.json --contamination-json all_evals_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_0.1_summary.jsonl --output-file decontamination_plot.pdf


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file-1", type=str)
    parser.add_argument("--json-file-2", type=str)
    parser.add_argument("--contamination-json", type=str)
    parser.add_argument("--contamination-summary", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--thresh", type=float, default=0.8)
    parser.add_argument("--eval-metadata", type=str, default="eval/eval_meta_data.csv")
    args = parser.parse_args(args)
    return args


def get_aggregated_results(data, data_control, eval_metadata):

    eval_metadata["results"] = eval_metadata["Eval Task"].map(data["eval_metrics"]["icl"])
    eval_metadata["results control"] = eval_metadata["Eval Task"].map(data_control["eval_metrics"]["icl"])
    eval_metadata["centered results"] = (
        eval_metadata["results"].astype(float) - 0.01 * eval_metadata["Random baseline"].astype(float)
    ) / (1.0 - 0.01 * eval_metadata["Random baseline"].astype(float))

    eval_metadata["centered results control"] = (
        eval_metadata["results control"].astype(float) - 0.01 * eval_metadata["Random baseline"].astype(float)
    ) / (1.0 - 0.01 * eval_metadata["Random baseline"].astype(float))

    eval_metadata["centered results diff"] = (
        eval_metadata["centered results"] - eval_metadata["centered results control"]
    )

    return eval_metadata


def main(args):
    args = parse_args(args)

    with open(args.json_file_1, "r") as f:
        data_1 = json.load(f)

    with open(args.json_file_2, "r") as f:
        data_2 = json.load(f)

    df = pd.read_json(args.contamination_json, lines=True)

    with open(args.contamination_summary, "r") as f:
        overlaps_summary = json.load(f)

    with open("eval/additional_aggregation.json", "r") as f:
        aggregation_json = json.load(f)

    data_1["eval_metrics"]["icl"]["mmlu"] = data_1["eval_metrics"]["icl"]["mmlu_fewshot"]
    data_2["eval_metrics"]["icl"]["mmlu"] = data_2["eval_metrics"]["icl"]["mmlu_fewshot"]

    overlaps_modified = {}
    for key in overlaps_summary.keys():
        chosen_df = df[df["eval task"] == key]
        total = len(chosen_df["overlap_counts"])
        contaminated = sum(chosen_df["overlap_counts"] > args.thresh)
        overlaps_modified[key] = 1.0 * contaminated / total

    overlaps_modified = {key[:-6]: value for key, value in overlaps_modified.items()}
    overlaps_modified["hellaswag_zeroshot"] = overlaps_modified["hellaswag"]
    overlaps_modified["jeopardy"] = overlaps_modified["jeopardy_all"]
    overlaps_modified["mmlu_fewshot"] = overlaps_modified["mmlu"]
    overlaps_modified["mmlu_zeroshot"] = overlaps_modified["mmlu"]
    overlaps_modified["winograd"] = overlaps_modified["winograd_wsc"]

    eval_metadata = pd.read_csv(args.eval_metadata)
    eval_metadata = get_aggregated_results(data_1, data_2, eval_metadata)
    eval_metadata["overlaps"] = eval_metadata["Eval Task"].map(overlaps_modified)

    # markers = ['o', 's', '^', 'x', '*', 'D', 'v', '+', 'p', 'h', '1', '2', '3', '4', '8', '|', '_', '.', ',', '<', '>', '^', 'v', 'd', 'H', 'P', 'X', 'o', 's', '^', 'x', '*', 'D', 'v', '+', 'p', 'h', '1', '2', '3', '4', '8', '|', '_', '.', ',', '<', '>', '^', 'v', 'd', 'H', 'P', 'X']

    markers = {
        "mmlu_fewshot": "x",
        "mmlu_zeroshot": "D",
        "winograd": "+",
        "squad": "^",
        "boolq": "s",
        "bigbench_operators": "h",
        "copa": "*",
    }

    colors = {
        "mmlu_fewshot": "k",
        "mmlu_zeroshot": "y",
        "winograd": "m",
        "squad": "c",
        "boolq": "g",
        "bigbench_operators": "r",
        "copa": "pink",
    }

    readable_names = {
        "mmlu_fewshot": "MMLU (5-shot)",
        "mmlu_zeroshot": "MMLU (0-shot)",
        "winograd": "Winograd",
        "squad": "SQuAD",
        "boolq": "BoolQ",
        "bigbench_operators": "BIG-bench (operators)",
        "copa": "COPA",
    }

    groups = eval_metadata.groupby("Eval Task")

    plt.figure(figsize=(6, 3), dpi=300)

    printed_label_other = False
    for i, (name, group) in enumerate(groups):
        if name in aggregation_json["low_variance_datasets"] and name not in markers:
            if np.isnan(group["overlaps"].values[0]):
                raise RuntimeError()

            label = None if printed_label_other else "Other Core datasets"
            plt.scatter(
                100.0 * group["overlaps"], 100.0 * group["centered results diff"], label=label, marker="o", color="blue"
            )
            printed_label_other = True

    for i, (name, group) in enumerate(groups):
        if name in markers:
            print(name, group["overlaps"].values[0])
            if np.isnan(group["overlaps"].values[0]):
                raise RuntimeError()
            plt.scatter(
                100.0 * group["overlaps"],
                100.0 * group["centered results diff"],
                label=readable_names[name],
                marker=markers[name],
                color=colors[name],
            )

    font_size = 15
    ax = plt.gca()
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    plt.xlabel("Percentage of evaluation set contaminated", fontsize=font_size)
    plt.ylabel("Centered acc. diff.", fontsize=font_size)
    plt.grid()
    # plt.legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim([-20.0, 13.0])
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=font_size)

    plt.savefig(args.output_file, bbox_inches="tight")


if __name__ == "__main__":
    main(sys.argv[1:])
