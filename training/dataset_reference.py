from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from typing import Dict, List, Union
import json


def replace_prefix(s3_url, prefix_replacement):
    if not prefix_replacement:
        return s3_url
    old_prefix, new_prefix = prefix_replacement.split("=")
    if s3_url.startswith(old_prefix):
        return s3_url.replace(old_prefix, new_prefix, 1)
    return s3_url


@dataclass
class DatasetReference:
    name: str
    sources: Union[str, List[str]]
    tokenized: bool
    num_tokens: int
    size: int
    dataset_url: str
    manifest_url: str
    dcnlp_commit_hash: str
    dcnlp_diff: str
    sampling_yaml: str

    uuid: str = uuid.uuid4().__str__()
    creation_date: datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    tokenizer: str = "EleutherAI/gpt-neox-20b"
    data_key: str = "json.gz"
    note: str = ""
    mirrors: Dict = None

    def update_for_mirror(self, mirror):
        if self.mirrors and mirror in self.mirrors:
            print(f"Updating dataset to use mirror {mirror}")
            for k, v in self.mirrors[mirror].items():
                previous_v = getattr(self, k, None)
                print(f"Updating {k} for mirror {mirror}: {previous_v} => {v}.")
                setattr(self, k, v)

    def replace_prefix(self, prefix_replacement):
        for k in ("dataset_url", "manifest_url"):
            new_url = replace_prefix(getattr(self, k), prefix_replacement)
            print(f"Replacing prefix in {k}: {getattr(self, k)} => {new_url}.")
            setattr(self, k, new_url)


# e.g.,

# dr = DatasetReference(
#     "rpj-pile-mix",
#     "",
#     True,
#     1_600_000_000_000,
#     -1,
#     [
#         "s3://***REMOVED***/rpj_tokenized_upsampled_eleutherai/manifest.jsonl",
#         "s3://***REMOVED***/2T_no_rpj_tokenized_upsampled_25k_shards/manifest.jsonl",
#     ],
#     [
#         0.725,
#         0.275,
#     ],
#     "",
#     "",
#     data_key="json",
# )

# dr = DatasetReference(
#     "rpj-pile-mix",
#     "",
#     True,
#     1_600_000_000_000,
#     -1,
#     [
#         "s3://***REMOVED***/rpj_tokenized_upsampled_eleutherai/manifest.jsonl",
#         "s3://***REMOVED***/2T_no_rpj_tokenized_upsampled_25k_shards/manifest.jsonl",
#     ],
#     [
#         0.725,
#         0.275,
#     ],
#     "",
#     "",
#     data_key="json",
# )


# print(json.dumps(asdict(dr), indent=4))
