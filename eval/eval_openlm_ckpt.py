import argparse
import builtins as __builtin__
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List


sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytz
import random
import torch
from aggregated_metrics import get_aggregated_results
from utils import update_args_from_openlm_config
from composer.loggers import InMemoryLogger, LoggerDestination
from composer.trainer import Trainer
from composer.utils import dist, get_device, reproducibility
from llmfoundry.utils.builders import build_icl_evaluators, build_logger
from omegaconf import OmegaConf as om
from open_lm.attention import ATTN_ACTIVATIONS, ATTN_SEQ_SCALARS
from open_lm.data import get_data
from open_lm.distributed import broadcast_object, init_distributed_device, is_master, world_info_from_env
from open_lm.evaluate import evaluate_loop
from open_lm.model import create_params
from open_lm.main import load_model
from open_lm.evaluate import evaluate_loop
from open_lm.file_utils import pt_load
from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
from pytz import timezone
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXTokenizerFast, LlamaTokenizerFast
from training.file_utils import download_val_data, get_downstream_task_name, load_ppl_yaml

builtin_print = __builtin__.print


def setup_for_distributed(is_master):
    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def convert_gpqa(gpqa_dir, outdir):
    os.makedirs(outdir, exist_ok=True)
    for filename in ["gpqa_main.csv", "gpqa_diamond.csv", "gpqa_extended.csv"]:
        inpath = os.path.join(gpqa_dir, filename)
        outpath = os.path.join(outdir, Path(inpath).with_suffix(".jsonl").name)
        with open(outpath, "w") as f:
            df = pd.read_csv(inpath)
            rng = random.Random(42)
            for i in range(df.shape[0]):
                question = df["Question"][i]
                choices = np.array(
                    [
                        df["Correct Answer"][i],
                        df["Incorrect Answer 1"][i],
                        df["Incorrect Answer 2"][i],
                        df["Incorrect Answer 3"][i],
                    ]
                )
                idx = list(range(4))
                rng.shuffle(idx)
                choices = choices[idx]
                gold = idx.index(0)
                data = {"query": question, "choices": choices.tolist(), "gold": gold}
                f.write(json.dumps(data))
                f.write("\n")

    return


def check_and_download_data():
    if not os.path.exists("local_data"):
        current_dir = os.path.dirname(os.path.realpath(__file__))

        if os.path.exists(f"{current_dir}/local_data"):
            shutil.copytree(f"{current_dir}/local_data", "local_data")
        else:
            if dist.get_global_rank() == 0:
                print("local_data folder does not exist. Running bash script...")
                script_path = os.path.join(current_dir, "download_eval_data.sh")

                subprocess.call([script_path])

            else:
                # Let other workers sleep a bit before barrier.
                time.sleep(10)
            dist.barrier()

    if not os.path.exists("gpqa_data"):
        repo_dir = os.path.dirname(os.path.realpath(__file__))
        if dist.get_global_rank() == 0:
            subprocess.run(
                [
                    "unzip",
                    "-P",
                    "deserted-untie-orchid",
                    os.path.join(repo_dir, "gpqa/dataset.zip"),
                    "-d",
                    "gpqa_data_orig/",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            convert_gpqa("gpqa_data_orig/dataset", "gpqa_data")
            shutil.rmtree("gpqa_data_orig")
        else:
            time.sleep(10)
        dist.barrier()

    print("Done downloading data.")
    return

@torch.no_grad()
def evaluate(model, tokenizer, cfg):
    cfg.dist_timeout = cfg.get("dist_timeout", 600.0)

    reproducibility.seed_all(cfg.seed)
    dist.initialize_dist(get_device(None), timeout=cfg.dist_timeout)
    setup_for_distributed(dist.get_global_rank() == 0)

    # Check if the data is downloaded, if not, download it.
    check_and_download_data()

    composer_model = SimpleComposerOpenLMCausalLM(model, tokenizer)

    icl_tasks_w_categories = list(
        filter(lambda x: 0 if "has_categories" not in x else x["has_categories"], cfg.icl_tasks)
    )
    icl_tasks_w_categories = list(map(lambda x: x["label"], icl_tasks_w_categories))

    cfg_icl_tasks = [om.to_container(i, resolve=True) for i in cfg.icl_tasks]
    evaluators, logger_keys = build_icl_evaluators(
        cfg_icl_tasks, tokenizer, cfg.max_seq_len, cfg.device_eval_batch_size
    )
    in_memory_logger = InMemoryLogger()  # track metrics in the in_memory_logger
    loggers: List[LoggerDestination] = [
        build_logger(name, logger_cfg) for name, logger_cfg in (cfg.get("loggers") or {}).items()
    ]
    loggers.append(in_memory_logger)

    fsdp_config = None
    fsdp_config = om.to_container(fsdp_config, resolve=True) if fsdp_config is not None else None

    load_path = cfg.get("load_path", None)

    trainer = Trainer(
        model=composer_model,
        loggers=loggers,
        precision=cfg.precision,
        fsdp_config=fsdp_config,  # type: ignore
        load_path=load_path,
        load_weights_only=True,
        progress_bar=False,
        log_to_console=True,
        dist_timeout=cfg.dist_timeout,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    a = time.time()
    trainer.eval(eval_dataloader=evaluators)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    b = time.time()

    print(f"Ran eval in: {b-a} seconds")

    performance_on_tasks = defaultdict(list)
    for key in logger_keys:
        if key in in_memory_logger.data:
            result = in_memory_logger.data[key][0][1].item()
            flag = True
            if len(icl_tasks_w_categories) > 0:
                for task in icl_tasks_w_categories:
                    if task in key:
                        performance_on_tasks[task].append(result)
                        flag = False
            if flag:
                performance_on_tasks[key].append(result)

    report_results = {}
    for task in performance_on_tasks:
        result = sum(performance_on_tasks[task]) / len(performance_on_tasks[task])
        if len(task.split("/")) > 1:
            label = task.split("/")[1]
            report_results[label] = result
        else:
            report_results[task] = result
    print(report_results)
    return report_results


def set_args_for_val(args, data, key):
    setattr(args, "val_data", data)
    setattr(args, "val_data_key", key)
    setattr(args, "squash_mask_left", True)
    setattr(args, "target_mask_individual", 50400)
    setattr(args, "target_mask_left", 50300)
    setattr(args, "val_seq_ci", True)
    setattr(args, "val_tok_ci", True)
    return args


def dump_or_update_output(args, local_rank, eval_metrics=None, helm_eval_metrics=None, helm_reference_uuid=None):
    date_format = "%Y_%m_%d-%H_%M_%S"
    date = datetime.now(tz=pytz.utc)
    date = date.astimezone(timezone("US/Pacific"))
    date = date.strftime(date_format)

    output = {
        "uuid": helm_reference_uuid if helm_reference_uuid else str(uuid.uuid4()),
        "model": args.model,
        "creation_date": date,
    }

    assert eval_metrics is not None or helm_eval_metrics is not None, "No eval metrics provided"

    if os.path.exists(args.output_file):
        with open(args.output_file) as f:
            output = json.load(f)

        output["update_date"] = date

    if eval_metrics is not None:
        output["name"] = str(args.eval_yaml)[:-5]
        output["eval_metrics"] = eval_metrics

        with open(args.additional_aggregation, "r") as f:
            aggregation_json = json.load(f)

        eval_metadata = pd.read_csv(args.eval_meta_data)

        output = get_aggregated_results(output, eval_metadata, aggregation_json)

    elif helm_eval_metrics is not None:
        output["helm_eval_name"] = str(args.run_spec)[:-5]
        output["helm_eval_metrics"] = helm_eval_metrics
        # UUID is passed into HELM is for separating different runs. Given the
        # uuid creation time, it might be different from the eval's uuid. This
        # helm_eval_metric is useful when viewing results in HELM's built-in
        # viewer.
        output["helm_reference_uuid"] = helm_reference_uuid

    print("Eval output: ")
    print(json.dumps(output, indent=4, sort_keys=True))
    if local_rank == 0:
        with open(args.output_file, "w") as f:
            json.dump(output, f, indent=4)


def main():
    """
    Usage:
    python eval_openlm_ckpt.py --checkpoint <path_to_openlm_checkpoint>  --model <name_of_model_config> --eval-yaml <path_to_eval_yaml> --tokenizer <tokenizer_name_or_path>
    example:
    cd eval
    python eval_openlm_ckpt.py --checkpoint ../checkpoints/llama2_7b.pt --model llama2_7b.json --eval-yaml in_memory_hf_eval.yaml --tokenizer <path_to_tokenizer>
    multi-gpu example:
    cd eval
    torchrun --nproc_per_node 3 eval_openlm_ckpt.py --checkpoint ../checkpoints/llama2_7b.pt --model llama2_7b.json --eval-yaml in_memory_hf_eval.yaml --tokenizer <path_to_tokenizer>

    torchrun --nproc_per_node 3 eval_openlm_ckpt.py --checkpoint checkpoint.pt --config params.txt

    helm_eval:
    Note: before running helm_eval, make sure you have helm installed:
        `pip install crfm-helm[scenarios,slurm,cleva,metrics]@git+https://github.com/stanford-crfm/helm.git@v0.5.1`
    By default it is multi-gpu on all gpus found by `torch.cuda.device_count()`. To run on single gpu, set num_gpus=1
    You should be able to run this with the same arguments as above, but with the addition of `--use-helm`

    multi-gpu example:
    cd eval
    python eval_openlm_ckpt.py --use-helm --checkpoint checkpoint.pt --config params.txt --run-spec helm_heavy_exhaustive.conf
    """
    parser = argparse.ArgumentParser()
    # Arguments that openlm requires when we call load_model
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility, when None, will use the seed from the eval config file.",
    )
    parser.add_argument("--fsdp", default=False, action="store_true")
    parser.add_argument("--distributed", default=True, action="store_true")
    parser.add_argument("--resume", default=None, type=str)

    # Argument for uploading results
    parser.add_argument("--remote-sync", type=str, default=None)
    parser.add_argument("--remote-sync-protocol", type=str, default="s3", choices=["s3", "fsspec"])

    parser.add_argument("--checkpoint", default=None, type=str, help="Path to checkpoint to evaluate.")
    parser.add_argument("--eval-yaml", type=str, default="light.yaml")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--config", default=None, type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--hf-model", default=None)
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument(
        "--use-temp-working-dir",
        action="store_true",
        help="Use a temporary working directory for the evaluation. removing it when done. "
        "This is required if you wish to run multiple evaluations with the same datasets"
        " in parallel on the same node.",
    )
    parser.add_argument(
        "--eval_meta_data", default=f"{os.path.dirname(__file__)}/eval_meta_data.csv", help="Eval meta data file"
    )
    parser.add_argument(
        "--preset-world-size",
        type=int,
        default=None,
        help="Explicitly set the world size. Useful in cases where a different number of gpus per node need to be used.",
    )
    parser.add_argument(
        "--additional_aggregation",
        default=f"{os.path.dirname(__file__)}/additional_aggregation.json",
        help="Eval aggregation file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="val data for perplexity calc",
    )
    parser.add_argument(
        "--moe-freq",
        type=int,
        default=0,
        help="if set > 0, we will add MoE layer to every moe_freq layer.",
    )
    parser.add_argument(
        "--moe-num-experts",
        type=int,
        default=None,
        help="Number of experts for MoE",
    )

    parser.add_argument(
        "--moe-weight-parallelism",
        action="store_true",
        help="Add weight parallelism to MoE",
    )

    parser.add_argument(
        "--moe-expert-model-parallelism",
        action="store_true",
        help="Add expert model parallelism to MoE",
    )

    parser.add_argument(
        "--moe-capacity-factor",
        type=float,
        default=1.25,
        help="MoE capacity factor",
    )

    parser.add_argument(
        "--moe-loss-weight",
        type=float,
        default=0.1,
        help="MoE loss weight",
    )
    parser.add_argument(
        "--moe-top-k",
        type=int,
        default=2,
        help="MoE top k experts",
    )
    parser.add_argument(
        "--attn-name",
        type=str,
        default="xformers_attn",
        choices=["xformers_attn", "torch_attn", "custom_attn"],
        help="type of attention to use",
    )
    parser.add_argument(
        "--attn-activation",
        type=str,
        default=None,
        choices=list(ATTN_ACTIVATIONS.keys()),
        help="activation to use with custom_attn",
    )
    parser.add_argument(
        "--attn-seq-scalar",
        type=str,
        default=None,
        choices=list(ATTN_SEQ_SCALARS.keys()),
        help="different ways to set L, where L^alpha divides attention logits post activation",
    )
    parser.add_argument(
        "--attn-seq-scalar-alpha",
        type=float,
        default=None,
        help="power alpha to raise L to, where L^alpha divides attention logits post activation",
    )
    parser.add_argument(
        "--val-max-pop-ci",
        default=None,
        action="store",
        type=int,
        help="when running CIs what is the maximum population size for the inner loop",
    )
    parser.add_argument(
        "--val-iter-ci",
        default=10_000,
        action="store",
        type=int,
        help="how many times to sample to construct the CI for the outer loop",
    )
    parser.add_argument("--averager-name", help="If specified, load this averager from checkpoint.")

    parser.add_argument("--donot-compute-perplexity", action="store_true")
    parser.add_argument("--compute-downstream-perplexity", action="store_true")
    parser.add_argument("--compute-paloma-perplexity", action="store_true")
    parser.add_argument("--force-xformers", action="store_true")

    # HELM args
    parser.add_argument("--use-helm", action="store_true", help="Use helm to eval models")
    parser.add_argument("--run-spec", type=str, default=None, help="Run spec file name.")
    parser.add_argument(
        "--parallelization-method",
        type=str,
        choices=["slurm", "gpu"],
        default="gpu",
        help="The parallelization method to use. Defaults to using gpus.",
    )
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to use.")

    parser.add_argument(
        "--experiment",
        type=str,
        default="default",
        help="Name of the experiment. Useful if not using model names to differentiate among experiments.",
    )
    parser.add_argument("--copy-model", action="store_true", help="Copy the model to the evaluation directory.")
    parser.add_argument(
        "--handle-existing-model",
        type=str,
        choices=["keep", "overwrite"],
        default="overwrite",
        help="How to handle existing model.",
    )
    parser.add_argument("--clear-cache", action="store_true", help="Clear the cache before running.")

    args = parser.parse_args()
    orig_seed = args.seed  # may be overridden by config file if it exists

    if args.use_helm:
        from eval_openlm_ckpt_helm import main

        if args.run_spec is None:
            args.run_spec = args.eval_yaml

        helm_reference_uuid = str(uuid.uuid4())
        if os.path.exists(args.output_file):
            with open(args.output_file) as f:
                output = json.load(f)
            if "uuid" in output:
                helm_reference_uuid = output["uuid"]
        args.uuid = helm_reference_uuid

        helm_eval_metrics = main(args, write_to_output_file=False)
        dump_or_update_output(args, 0, helm_eval_metrics=helm_eval_metrics, helm_reference_uuid=helm_reference_uuid)
        return

    if args.config is not None:
        assert args.hf_model is None, (
            "If you are using a config file, "
            "you are trying to evaluate open_lm model. Please remove hf-model argument."
        )

        update_args_from_openlm_config(args)
        # disable wandb for eval
        args.wandb = None
    else:
        # Most probably evaling a hf-model.

        assert args.hf_model, (
            "If you are not using a config file, you might want to evaluate a Hugginface model, "
            "so please provide hf-model argument."
        )
        # Computing perplexity for HF model doesn't make sense.
        args.donot_compute_perplexity = True

        # Setting those params as they are needed to distributed evals
        # and they are supposed to come from config file.
        args.dist_backend = "nccl"
        args.dist_url = "env://"
        args.no_set_device_rank = False
        args.model = args.hf_model
        args.force_distributed = False
    with open(args.eval_yaml) as f:
        eval_cfg = om.load(f)
    if orig_seed is not None:
        print(f"Overriding eval config seed ({eval_cfg.seed}) to {orig_seed}")
        eval_cfg.seed = orig_seed

        # now need to set the 'fewshot_random_seed' in each config in the icl task configs
        for icl_cfg in eval_cfg.icl_tasks:
            icl_cfg.fewshot_random_seed = orig_seed

    args.resume = args.checkpoint
    args.remote_sync = args.output_file
    directory = os.path.dirname(args.output_file)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    CWD = os.getcwd()
    if args.use_temp_working_dir:
        temp_dir = os.path.join(CWD, "eval_openlm_ckpt_temp_dirs", f"{uuid.uuid4()}")
        os.makedirs(temp_dir, exist_ok=True)  # in case rank > 0
        os.chdir(temp_dir)
        print(f"Using temporary working directory: {temp_dir}")

    print("Loading model into the right classes")
    if args.hf_model is not None:
        eval_model = AutoModelForCausalLM.from_pretrained(
            args.hf_model, trust_remote_code=True, cache_dir=args.hf_cache_dir
        )
    else:
        params = create_params(args)
        eval_model = OpenLMforCausalLM(OpenLMConfig(params))

    if "gpt-neox-20b" in args.tokenizer:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    elif "llama" in args.tokenizer:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer)
        if len(tokenizer) > eval_model.config.vocab_size:  # happens in llama-3-8b
            print(f"Resizing vocab from {eval_model.config.vocab_size} to {len(tokenizer)}")
            eval_model.resize_token_embeddings(len(tokenizer))
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, cache_dir=args.hf_cache_dir)
    print(tokenizer)

    if args.checkpoint is not None:
        if not args.averager_name:
            print(f"Loading checkpoint {args.checkpoint}")
            args.distributed = False
            load_model(args, eval_model.model, different_seed=True)
            args.distributed = True
        else:
            print(f"Loading checkpoint {args.checkpoint}")
            checkpoint = pt_load(args.resume, map_location="cpu")
            if "epoch" in checkpoint:
                # resuming a train checkpoint w/ epoch and optimizer state
                start_epoch = checkpoint["epoch"]
                avg_sd = torch.load(args.checkpoint, map_location="cpu")
                if next(iter(avg_sd.items()))[0].startswith("module"):
                    avg_sd = {k[len("module.") :]: v for k, v in avg_sd.items()}
                eval_model.model.load_state_dict(avg_sd)

        # HF model loaded with from_pretrained is by default in eval mode.
        # https://github.com/huggingface/transformers/blob/ebfdb9ca62205279d5019ef1403877461b3b2da4/src/transformers/modeling_utils.py#L2500
        eval_model.model.eval()

    # Set requires grad = False to reduce memory consumption - o/w composer makes a copy of the model.
    for p in eval_model.parameters():
        p.requires_grad = False

    device = init_distributed_device(args)
    eval_model = eval_model.to(device)
    eval_metrics = {}

    local_rank, _, _ = world_info_from_env()

    if not args.donot_compute_perplexity:
        args.per_gpu_val_batch_size = args.per_gpu_batch_size // args.accum_freq
        openlm_val_data = download_val_data("open_lm_val", skip_download=local_rank != 0)
        args = set_args_for_val(args, [openlm_val_data], ["json"])
        data = get_data(args, epoch=0, tokenizer=None, skip_train=True)
        results = evaluate_loop(eval_model.model, data["val_list"], 0, args, None)
        perplexity_val = results[0]["loss"]
        eval_metrics["perplexity"] = perplexity_val

    if args.compute_paloma_perplexity:
        args.per_gpu_val_batch_size = args.per_gpu_batch_size // args.accum_freq
        paloma_val_data = download_val_data("paloma_val", skip_download=local_rank != 0)
        args = set_args_for_val(args, [paloma_val_data], ["json.gz"])
        data = get_data(args, epoch=0, tokenizer=None, skip_train=True)
        results = evaluate_loop(eval_model.model, data["val_list"], 0, args, None)
        perplexity_val = results[0]["loss"]
        eval_metrics["paloma_perplexity"] = perplexity_val

    if args.compute_downstream_perplexity:
        args.per_gpu_val_batch_size = args.per_gpu_batch_size // args.accum_freq
        size = args.eval_yaml[:-5]
        tasks = load_ppl_yaml(size)
        downstream_datas = [
            download_val_data(task_name, skip_download=local_rank != 0)
            for task_name in tasks
            if "gpqa" not in task_name
        ]
        args = set_args_for_val(args, downstream_datas, ["txt"] * len(downstream_datas))
        data = get_data(args, epoch=0, tokenizer=None, skip_train=True)
        results = evaluate_loop(eval_model.model, data["val_list"], 0, args, None)
        eval_metrics["downstream_perpexity"] = {}
        for result in results:
            data_name = result["val_data"][0].split("/")[-2]
            eval_metrics["downstream_perpexity"][data_name] = result["loss"]

    icl_results = evaluate(eval_model, tokenizer, eval_cfg)
    eval_metrics["icl"] = icl_results

    dump_or_update_output(args, local_rank, eval_metrics=eval_metrics)


if __name__ == "__main__":
    main()
