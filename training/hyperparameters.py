from pathlib import Path
import json
import os
from dataclasses import dataclass
from typing import List


def sanitize_for_fs(s):
    return str(s).replace(".", "p").replace("/", "-")


@dataclass
class Hyperparameters:
    model: str
    tokens: int
    warmup: int
    lr: float
    wd: float
    cd: float
    global_bs: int
    acc: int
    qk_norm: bool
    z_loss: float
    grad_checkpointing: bool
    params: int
    params_no_embed: int
    fsdp_flags: List[str]
    chinchilla_multiplier: float
    seed: int = 124
    vocab_size: int = 50432
    norm: str = "gain_only_lp_layer_norm"

    def update_config(self, args):
        if args.warmup is not None:
            self.warmup = args.warmup

        if args.lr is not None:
            self.lr = args.lr

        if args.wd is not None:
            self.wd = args.wd

        if args.cd is not None:
            self.cd = args.cd

        if args.global_bs is not None:
            self.global_bs = args.global_bs

        if args.acc is not None:
            self.acc = args.acc

        if args.chinchilla_multiplier is not None:
            self.chinchilla_multiplier = args.chinchilla_multiplier

        self.tokens = int(self.tokens * self.chinchilla_multiplier)
        self.seed = args.seed

    def get_friendly_name(self, data, suffix=None):
        data_n = data.name
        model_n = self.model.split("/")[-1].split(".")[0]
        w_n = f"warm={sanitize_for_fs(self.warmup)}"
        lr_n = f"lr={sanitize_for_fs(self.lr)}"
        wd_n = f"wd={sanitize_for_fs(self.wd)}"
        cd_n = f"cd={sanitize_for_fs(self.cd)}"
        bs_n = f"bs={sanitize_for_fs(self.global_bs)}"
        cc_n = f"mult={sanitize_for_fs(self.chinchilla_multiplier)}"
        seed_n = f"seed={sanitize_for_fs(self.seed)}"
        tokens_n = f"tokens={sanitize_for_fs(self.tokens)}"

        name = f"{data_n}-{model_n}-{w_n}-{lr_n}-{wd_n}-{cd_n}-{bs_n}-{cc_n}-{seed_n}-{tokens_n}"
        if self.norm != Hyperparameters.norm:  # Add only if not default
            name = f"{name}-norm={sanitize_for_fs(self.norm)}"
        if self.vocab_size != Hyperparameters.vocab_size:  # Add only if not default
            name = f"{name}-vocab={sanitize_for_fs(self.vocab_size)}"

        if suffix:
            name = f"{name}{suffix}"
        return name


SCALE_CONFIG_PATHS = [
    Path(__file__).parent / f"configs/",
    Path(__file__).parent / f"configs_ppl_filtering/",
    Path(__file__).parent / f"configs_grid/",
]
SCALE_CONFIGS = {}

for s in SCALE_CONFIG_PATHS:
    if not os.path.isdir(s):
        continue
    for p in os.listdir(s):
        with open(os.path.join(s, p), "r") as f:
            SCALE_CONFIGS[Path(p).stem] = Hyperparameters(**json.load(f))


def available_scales(simple_names=False):
    return sorted(list(SCALE_CONFIGS.keys()))


def get_scale_config(scale):
    if scale not in SCALE_CONFIGS:
        raise ValueError(f"Unknown scale: {scale}. Please use one of {available_scales()}")
    return SCALE_CONFIGS[scale]
