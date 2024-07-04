from .dedup_jsonl import dedup_jsonl
from baselines.core.constants import GLOBAL_FUNCTIONS

GLOBAL_FUNCTIONS["exact_dedup"] = dedup_jsonl
