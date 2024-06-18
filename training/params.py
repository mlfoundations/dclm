import argparse
import json
import os
from pathlib import Path

import torch.distributed as dist
from open_lm.distributed import world_info_from_env

from training.file_utils import download_val_data, load_ppl_yaml, tok_mult_paths
from training.hyperparameters import available_scales


def add_dcnlp_args(parser):
    parser.add_argument(
        "--scale",
        type=str,
        required=False,
        default=None,
        choices=available_scales(),
        help="Competition scale.",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        required=False,
        default=None,
        help="Competition scale.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        required=False,
        default=None,
        help="Place to write logs and checkpoints.",
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        required=False,
        help="If set, experiments will be sync'd to s3.",
        default=None,
    )
    parser.add_argument(
        "--clean-exp",
        default=False,
        action="store_true",
        help="If set, local exp dir will be cleared, only supported if --remote-sync specified.",
    )
    parser.add_argument("--git-db", default="exp_data/models", type=str, help="place to save output model.json")

    parser.add_argument("--workers", type=int, default=2, help="Number of workers for open_lm.")
    parser.add_argument(
        "--precision",
        type=str,
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"],
        default="amp_bfloat16",
        help="Floating point precision.",
    )
    parser.add_argument(
        "--num-checkpoints",
        type=int,
        default=5,
        help="Number of times we save checkpoints during training.",
    )
    parser.add_argument("--seed", type=int, default=124, help="Random seed.")
    parser.add_argument(
        "--report-to-wandb",
        default=False,
        action="store_true",
        help="If True, report to wandb.",
    )
    parser.add_argument(
        "--val-frequency",
        type=int,
        default=5,
        help="How often to run evaluation with val-data (in epochs). Last epoch validated if val-data provided.",
    )
    parser.add_argument(
        "--manifest-prefix-override",
        type=str,
        required=False,
        default=None,
        help="Overide the manifest prefix for the target dataset.json",
    )
    parser.add_argument("--prefix-replacement", default="", help="Prefix replacement in S3 URL")
    parser.add_argument(
        "--remote-sync-override",
        type=str,
        required=False,
        default=None,
        help="Overide the manifest prefix for the target dataset.json",
    )
    parser.add_argument(
        "--chinchilla-multiplier",
        type=float,
        required=False,
        help="Support token multiplier.",
    )
    parser.add_argument(
        "--re-evaluate",
        required=False,
        type=str,
        default=None,
        help="pass a model json",
    )
    parser.add_argument(
        "--do-eval",
        action="store_true",
        help="Whether to skip evaluation",
    )
    parser.add_argument(
        "--downstream-eval",
        required=False,
        action="store_true",
        help="Eval on llm-foundry as a loss evaluation.",
    )
    parser.add_argument(
        "--tokmult-eval",
        required=False,
        action="store_true",
        help="Eval on eval sets for tokmult paper.",
    )

    # override hparams in the hparam config provided in --scale (good for grid searches)
    parser.add_argument(
        "--warmup",
        required=False,
        type=int,
        default=None,
        help="number of warmup steps",
    )
    parser.add_argument(
        "--lr",
        required=False,
        type=float,
        default=None,
        help="max lr for training",
    )
    parser.add_argument(
        "--wd",
        required=False,
        type=float,
        default=None,
        help="weight-decay for training",
    )
    parser.add_argument(
        "--cd",
        required=False,
        type=float,
        default=None,
        help="cool down ending lr",
    )
    parser.add_argument(
        "--global-bs",
        required=False,
        type=int,
        default=None,
        help="global batch size",
    )
    parser.add_argument(
        "--acc",
        required=False,
        type=int,
        default=None,
        help="gradient acc steps",
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="If true, skip training. Useful for creating a model json."
    )
    parser.add_argument("--pretrained", type=str, default=None, help="Checkpoint to start model from.")
    parser.add_argument(
        "--load-pretrained-state",
        action="store_true",
        help="Whether to resume the optimizer from a pretrained model. Default is false. Only relevant when --pretrained is not None.",
    )
    parser.add_argument("--multiple-data-passes", action="store_true")
    parser.add_argument("--mirror", help="Use this dataset mirror if it exists in the dataset 'mirrors' key.")
    parser.add_argument("--name-suffix", help="Suffix to append to the friendly name")
    parser.add_argument(
        "--partial-model-dir",
        type=str,
        default=None,
        help="If not None, will save intermediate model jsons for partial evals.",
    )
    parser.add_argument("--attn-name", default="auto", type=str, help="attention")
    parser.add_argument("--torchcompile", action="store_true", help="Use torchcompile.")
    parser.add_argument(
        "--averagers",
        type=str,
        default=None,
        help="Optinoally average checkpoints along the trajectory.",
    )
    parser.add_argument(
        "--log-avg-model-training-loss",
        type=int,
        default=0,
        help="Whether to log the average model training loss. if not 0, it will log the average loss over the specified number of steps.",
    )
    parser.add_argument(
        "--data-tolerate-error-p",
        type=float,
        default=0.09,  # Roughly the number required to not repeat more than 10% of data.
        help="This is the percentage of expected tokens above which the checkpoint is considered failed because of not having seen enough data.",
    )
    parser.add_argument(
        "--data-tolerate-num-ckpts",
        type=int,
        default=0,
        help="This is the maximum number of failed checkpoints (due to not having seen enough tokens) that are allowed",
    )


def parse_dcnlp_args():
    parser = argparse.ArgumentParser()
    add_dcnlp_args(parser)
    args = parser.parse_args()

    if args.re_evaluate is not None:
        # in the case of re-evaluation set based on the input model.json
        model_json = None
        with open(args.re_evaluate, "r") as f:
            model_json = json.load(f)

        args.scale = Path(model_json["hyperparameters"]["model"]).stem
        args.data_config = f"exp_data/datasets/tokenized/{model_json['dataset_name']}.json"

        for i in range(len(model_json["open_lm_args"])):
            if model_json["open_lm_args"][i] == "--logs":
                args.logs = model_json["open_lm_args"][i + 1]
            if model_json["open_lm_args"][i] == "--remote-sync":
                if args.remote_sync_override:
                    args.remote_sync = args.remote_sync_override
                else:
                    args.remote_sync = model_json["open_lm_args"][i + 1]

    assert args.scale is not None and args.data_config is not None and args.logs is not None
    assert not (args.tokmult_eval and args.downstream_eval)

    return args


def get_open_lm_args(args, hparams, dr):
    if args.manifest_prefix_override is not None:
        assert args.prefix_replacement == ""
        manifest_name = Path(dr.manifest_url).name
        dr.manifest_url = os.path.join(args.manifest_prefix_override, f"{manifest_name}")

    if args.mirror:
        dr.update_for_mirror(args.mirror)

    if args.prefix_replacement:
        assert args.manifest_prefix_override is None
        dr.replace_prefix(args.prefix_replacement)

    local_rank, _, _ = world_info_from_env()

    open_lm_args = [
        "--workers",
        f"{args.workers}",
        "--precision",
        args.precision,
        "--global-batch-size",
        f"{hparams.global_bs}",
        "--log-every-n-steps",
        "20",
        "--grad-clip-norm",
        "1",
        "--lr",
        f"{hparams.lr}",
        "--warmup",
        f"{hparams.warmup}",
        "--model",
        f"{hparams.model}",
        "--wd",
        f"{hparams.wd}",
        "--beta2",
        "0.95",
        "--epochs",
        f"{args.num_checkpoints}",
        "--resume",
        "latest",
        "--seed",
        f"{args.seed}",
        "--accum-freq",
        f"{hparams.acc}",
        "--model-norm",
        hparams.norm,
        "--delete-previous-checkpoint",
        "--lr-cooldown-end",
        f"{hparams.cd}",
        "--logs",
        f"{args.logs}",
        "--attn-name",
        f"{args.attn_name}",
        "--log-logit-mean",
        "--data-tolerate-error-p",
        f"{args.data_tolerate_error_p}",
        "--data-tolerate-num-ckpts",
        f"{args.data_tolerate_num_ckpts}",
    ]

    if args.pretrained is not None:
        open_lm_args.extend(["--pretrained", f"{args.pretrained}"])
        if args.load_pretrained_state:
            open_lm_args.extend(["--load-pretrained-state"])

    if args.multiple_data_passes:
        open_lm_args.append("--multiple-data-passes")

    if args.averagers:
        open_lm_args.extend(["--averagers", args.averagers])
        open_lm_args.extend(["--log-avg-model-training-loss", str(args.log_avg_model_training_loss)])

    name = None
    if args.re_evaluate is None:
        # case where we are training
        name = hparams.get_friendly_name(dr, args.name_suffix)

        open_lm_args.extend(
            [
                "--train-num-samples",
                f"{hparams.tokens // args.num_checkpoints}",
                "--dataset-manifest",
                dr.manifest_url,
                "--data-key",
                dr.data_key,
                "--name",
                name,
            ]
        )
        # add fsdp flags that are different for "smaller" and "larger" configs

        open_lm_args.extend(hparams.fsdp_flags)
    else:
        # get name from the passed model.json
        model_json = None
        with open(args.re_evaluate, "r") as f:
            model_json = json.load(f)

        name = model_json["name"]
        open_lm_args.extend(
            [
                "--name",
                name,
            ]
        )

    if args.do_eval:
        openlm_val_data = download_val_data("open_lm_val", skip_download=local_rank != 0)
        c4_val_data = download_val_data("c4_val", skip_download=local_rank != 0)
        paloma_val_data = download_val_data("paloma_val", skip_download=local_rank != 0)

    if not args.do_eval:
        pass
    elif args.downstream_eval:
        tasks = load_ppl_yaml()
        downstream_datas = [download_val_data(task_name, skip_download=local_rank != 0) for task_name in tasks]

        open_lm_args.extend(
            [
                "--val-data",
                openlm_val_data,
                c4_val_data,
                # paloma_val_data,
                *downstream_datas,
                "--val-frequency",
                f"{args.val_frequency}",
                "--val-data-key",
                "json",
                "txt",
                # "json.gz",
                *(["txt"] * len(downstream_datas)),
                "--squash-mask-left",
                "--target-mask-individual",
                "50400",
                "--target-mask-left",
                "50300",
                "--val-tok-ci",
                "--val-seq-ci",
                "--val-max-pop-ci",
                "300000",
            ]
        )
    elif args.tokmult_eval:
        tasks = load_ppl_yaml()
        downstream_datas = [download_val_data(task_name, skip_download=local_rank != 0) for task_name in tasks]

        open_lm_args.extend(
            [
                "--val-data",
                *tok_mult_paths,
                *downstream_datas,
                "--val-frequency",
                f"{args.val_frequency}",
                "--val-data-key",
                "json",  # hack open_lm, c4_val are the two entry in tok_mult_paths
                "txt",
                # "json.gz",
                *(["json.gz"] * (len(tok_mult_paths) - 2)),
                *(["txt"] * len(downstream_datas)),
                "--squash-mask-left",
                "--target-mask-individual",
                "50400",
                "--target-mask-left",
                "50300",
                "--val-tok-ci",
                "--val-seq-ci",
                "--val-max-pop-ci",
                "300000",
            ]
        )
    else:
        open_lm_args.extend(
            [
                "--val-data",
                openlm_val_data,
                c4_val_data,
                # paloma_val_data,
                "--val-frequency",
                f"{args.val_frequency}",
                "--val-data-key",
                "json",
                "txt",
                # "json.gz",
                "--val-tok-ci",
                "--val-seq-ci",
                "--val-max-pop-ci",
                "300000",
            ]
        )

    if args.report_to_wandb:
        open_lm_args.extend(["--report-to", "wandb", "--wandb-project-name", "dcnlp"])

    if hparams.qk_norm:
        open_lm_args.append("--qk-norm")
    if hparams.grad_checkpointing:
        open_lm_args.append("--grad-checkpointing")
    if hparams.z_loss > 0:
        open_lm_args.extend(["--z-loss", f"{hparams.z_loss}"])
    if args.remote_sync:
        open_lm_args.extend(
            [
                "--remote-sync",
                f"{args.remote_sync}",
            ]
        )
    if args.torchcompile:
        open_lm_args.append("--torchcompile")

    return open_lm_args, name
