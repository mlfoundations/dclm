import json
import argparse
import pandas as pd
import os

CURRENT_VERSION = "v2"

def gen_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_meta_data", default=f"{os.path.dirname(__file__)}/eval_meta_data.csv", help="Eval meta data file"
    )
    parser.add_argument(
        "--additional_aggregation",
        default=f"{os.path.dirname(__file__)}/additional_aggregation.json",
        help="Eval aggregation file",
    )
    parser.add_argument("--eval_results", help="Eval results")
    parser.add_argument("--version", default=CURRENT_VERSION, help="Version of the evaluation results. Do not change unlesss you want the older (incorrect) centered averages.")
    return parser


def get_aggregated_results(data, eval_metadata, aggregation_json, version=CURRENT_VERSION):

    data["missing tasks"] = str(
        [task for task in eval_metadata["Eval Task"] if task not in data["eval_metrics"]["icl"]]
    )
    eval_metadata["results"] = eval_metadata["Eval Task"].map(data["eval_metrics"]["icl"])
    eval_metadata["centered results"] = (
        eval_metadata["results"].astype(float) - 0.01 * eval_metadata["Random baseline"].astype(float)
    ) / (1.0 - 0.01 * eval_metadata["Random baseline"].astype(float))
    result_df = eval_metadata.groupby("Task Category").agg({"centered results": "mean"}).reset_index()
    data["aggregated_task_categories_centered"] = result_df.set_index("Task Category").to_dict()["centered results"]
    data["aggregated_centered_results"] = eval_metadata["centered results"].mean()
    data["aggregated_results"] = eval_metadata["results"].mean()

    for key in aggregation_json:
        tasks = aggregation_json[key]
        data[key] = eval_metadata[eval_metadata["Eval Task"].isin(tasks)]["results"].mean()
        data[f"{key}_centered"] = eval_metadata[eval_metadata["Eval Task"].isin(tasks)]["centered results"].mean()

    # add the new names
    if 'low_variance_datasets_centered' in data:
        # missing task for Core:
        missing_tasks_for_core = [task for task in aggregation_json['low_variance_datasets']
                                  if task not in data["eval_metrics"]["icl"]]
        if missing_tasks_for_core:
            data[f'Core_{version}'] = "N/A due to missing tasks: " + str(missing_tasks_for_core)
        else:
            data[f'Core_{version}'] = data['low_variance_datasets_centered']
    if 'aggregated_centered_results' in data:
        if data["missing tasks"] != "[]":
            data[f'Extended_{version}'] = "N/A due to missing tasks: " + data["missing tasks"]
        else:
            data[f'Extended_{version}'] = data['aggregated_centered_results']

    data['eval_version'] = version

    # Handle migration for old results
    if version == CURRENT_VERSION:
        if 'Core' in data and 'Core_v1' not in data:
            # If updating an older results file, migrate CORE --> Core_v1 (which won't be present)
            data['Core_v1'] = data['Core']
            del data['Core']
        if 'Extended' in data and 'Extended_v1' not in data:
            # If updating an older results file, migrate Extended --> Extended_v1 (which won't be present)
            data['Extended_v1'] = data['Extended']
            del data['Extended']
    
    # Set unversioned keys to point to current version if it is present
    if f'Core_{CURRENT_VERSION}' in data:
        data['Core'] = data[f'Core_{CURRENT_VERSION}'] 
    if f'Extended_{CURRENT_VERSION}' in data:
        data['Extended'] = data[f'Extended_{CURRENT_VERSION}']        
    return data


def main():
    parser = gen_parser()
    args = parser.parse_args()

    if args.version == CURRENT_VERSION:
        args.eval_meta_data = f"{os.path.dirname(__file__)}/eval_meta_data.csv"
    elif args.version == "v1":
        args.eval_meta_data = f"{os.path.dirname(__file__)}/eval_meta_data_v1.csv"

    eval_metadata = pd.read_csv(args.eval_meta_data)

    with open(args.eval_results, "r") as f:
        data = json.load(f)

    with open(args.additional_aggregation, "r") as f:
        aggregation_json = json.load(f)

    data = get_aggregated_results(data, eval_metadata, aggregation_json, version=args.version)

    with open(args.eval_results, "w") as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    main()
