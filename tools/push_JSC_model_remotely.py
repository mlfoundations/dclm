import json
import boto3
import argparse
from botocore.exceptions import NoCredentialsError
from pathlib import Path

# Copies models from local JSC location to remote bucket and changes the model json files


def upload_file_to_s3(file_name, bucket, object_name=None):
    """
    Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified, file_name is used
    :return: True if file was uploaded, else False
    """
    if object_name is None:
        object_name = file_name

    s3_client = boto3.client("s3")

    print(f"Uploading {file_name} to bucket={bucket} at {object_name}")
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except NoCredentialsError:
        print("Credentials not available")
        return False
    except Exception as e:
        print(e)
        return False
    return True


parser = argparse.ArgumentParser(description="JSC push models and update json")

# Add an argument for the JSON file path
parser.add_argument("--json_path", type=str, help="Path to the JSON file")
parser.add_argument("--s3_path", type=str, help="Path to the s3 bucket for model")

args = parser.parse_args()

BUCKET_PATH = args.s3_path  # s3://dcnlp-west/dsir_wiki/experiments/models/CC_v3_random_resampled_3B-open_lm_160m-1.0/
LOCAL_PATH = (
    args.json_path
)  # /p/home/jusers/garg4/juwels/garg4/scratch/code/dcnlp/exp_data/models/CC_v3_random_resampled_3B-open_lm_160m-1.0.json

with open(LOCAL_PATH, "r") as f:
    data = json.load(f)

local_model_path = Path(data["checkpoint_url"])
local_param_path = Path(data["params_url"])

bucket_name = BUCKET_PATH.split("/")[2]
remote_model_path = Path("/".join(BUCKET_PATH.split("/")[3:])) / local_model_path.name
remote_param_path = Path("/".join(BUCKET_PATH.split("/")[3:])) / local_param_path.name

# Example usage
upload_file_to_s3(local_model_path, bucket_name, str(remote_model_path))
upload_file_to_s3(local_param_path, bucket_name, str(remote_param_path))

data["checkpoint_url"] = "s3://" + str(f"{bucket_name}" / remote_model_path)
data["params_url"] = "s3://" + str(f"{bucket_name}" / remote_param_path)

with open(LOCAL_PATH, "w") as f:
    json.dump(data, f, indent=4)
