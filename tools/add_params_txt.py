import argparse
import json


def modify_json(file_path):
    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Extract the checkpoint_url
    checkpoint_url = data.get("checkpoint_url")
    if checkpoint_url:
        # Create the new params_url
        params_url = "/".join(checkpoint_url.split("/")[:-2]) + "/params.txt"
        data["params_url"] = params_url

        # Save the modified JSON back to the file
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

        print(f"Modified JSON data with params_url: {params_url}")
    else:
        print("No 'checkpoint_url' key found in JSON.")


def main():
    parser = argparse.ArgumentParser(description="Modify a JSON file to add a params_url key.")
    parser.add_argument("file_path", help="Path to the JSON file")

    args = parser.parse_args()
    modify_json(args.file_path)


if __name__ == "__main__":
    main()
