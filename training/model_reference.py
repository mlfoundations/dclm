import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List

from training.hyperparameters import Hyperparameters


@dataclass
class ModelReference:
    name: str
    dataset_name: str
    dataset_uuid: str
    hyperparameters: Hyperparameters
    checkpoint_url: str
    open_lm_version: str
    open_lm_args: str
    results: List[Any]
    params_url: str

    uuid: str = uuid.uuid4().__str__()
    creation_date: datetime = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    failed: bool = False
    error: str = ""
