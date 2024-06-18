import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import fsspec
import torch
from open_lm.distributed import world_info_from_env

from training.dataset_reference import DatasetReference
from training.file_utils import natural_key, setup_logger, start_partial_model_process, terminate_partial_model_process
from training.hyperparameters import Hyperparameters, get_scale_config
from training.model_reference import ModelReference
from training.params import get_open_lm_args, parse_dcnlp_args

logger = setup_logger(__name__)


def process_dcnlp_args(args):
    """Helper script for setting up data reference, hparams, and name.

    Note: The reason this is a function is because it is used by other scripts (e.g. Sagemaker) to get the name from an
    args object.
    """
    data = None
    with open(args.data_config, "r") as f:
        data = DatasetReference(**json.load(f))

    # modify num tokens by multiplier
    hparams = None
    if args.re_evaluate:
        model_json = None
        with open(args.re_evaluate, "r") as f:
            model_json = json.load(f)
        hparams = Hyperparameters(**model_json["hyperparameters"])
        hparams.global_bs = 128
    else:
        hparams = get_scale_config(args.scale)

        # if argparse overrides scale config we should too
        # NOTE: this will be removed for public release but useful for grid search
        hparams.update_config(args)

    open_lm_args, name = get_open_lm_args(args, hparams, data)
    return open_lm_args, name, hparams, data


if __name__ == "__main__":
    args = parse_dcnlp_args()

    if args.clean_exp:
        assert args.remote_sync is not None, "must specify --remote-sync to use --clean-local-logs"

    open_lm_args, name, hparams, data = process_dcnlp_args(args)

    _, rank, world_size = world_info_from_env()
    if rank == 0:
        logger.info(f"Running training on scale: {args.scale}")
        logger.info(f"World size is {world_size}.")

    assert (
        hparams.global_bs % world_size == 0
    ), f"world size: {world_size} does not divide global batch size: {hparams.global_bs}"

    exp_data_models_path = Path(__file__).parent.parent / args.git_db
    if not exp_data_models_path.exists():
        os.makedirs(exp_data_models_path, exist_ok=True)

    model_path = exp_data_models_path / f"{name}.json"

    if os.path.exists(model_path) and not args.re_evaluate:
        if rank == 0:
            logger.info(f"{model_path} already exists, please manually delete it to run a fresh training.")
        exit(0)

    if not os.path.exists(os.path.join(args.logs, name)):
        # create this dir to prevent sync'ing errors
        os.makedirs(os.path.join(args.logs, name), exist_ok=True)

    if args.partial_model_dir is not None:
        if rank == 0:
            os.makedirs(args.partial_model_dir, exist_ok=True)
            root_path = args.remote_sync if args.remote_sync is not None else args.logs
            partial_model_proc = start_partial_model_process(
                args.partial_model_dir, os.path.join(root_path, name), name, data, hparams, open_lm_args
            )
            partial_model_proc.start()
    else:
        partial_model_proc = None

    if not args.skip_train:
        from open_lm.main import main

        print(f"Running with args:\n{open_lm_args}")

        try:
            main(open_lm_args)
        except Exception as e:
            if rank == 0:
                logger.error(e)
                logger.error(traceback.format_exc())

    if rank == 0:
        time.sleep(10)
        if partial_model_proc is not None:
            terminate_partial_model_process(partial_model_proc)
        fs, exp_root = None, None
        if args.remote_sync:
            fs, exp_root = fsspec.core.url_to_fs(os.path.join(args.remote_sync, name))
        else:
            fs, exp_root = fsspec.core.url_to_fs(os.path.join(args.logs, name))

        results_jsonl = os.path.join(exp_root, "checkpoints", "results.jsonl")

        if exp_root.startswith("s3://"):
            result = subprocess.run(
                ["aws", "s3", "ls", os.path.join(exp_root, "checkpoints"), "--recursive"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            files = [l.strip().split(" ")[-1] for l in result.stdout.decode().splitlines()]
            stats = [os.path.join(exp_root, "checkpoints", Path(f).name) for f in files if "stats" in Path(f).name]
        else:
            stats_glob = os.path.join(exp_root, "checkpoints", "stats_*.pt")
            stats = fs.glob(stats_glob)

        stats = sorted(stats, key=natural_key)
        if not stats:
            raise ValueError(f"Could not find stats in {stats_glob}")
        final_stats = stats[-1]

        # check stats file to make sure that this is in fact the last epoch
        stats = None
        remote_model_path = os.path.join(exp_root, f"{name}.json")
        with fs.open(final_stats, "rb") as f:
            stats = torch.load(f)
        try:
            assert stats["is_final_checkpoint"], "not final checkpoint, training exited early"

            final_checkpoint = final_stats.replace("stats", "epoch")
            params_txt = os.path.join(exp_root, "params.txt")

            assert fs.exists(final_checkpoint), f"final checkpoint does not exist at {final_checkpoint}"
            assert fs.exists(params_txt), f"params.txt does not exist at {params_txt}"

            logger.info("Done training.")

            model = None
            if args.re_evaluate:
                read_data = None
                with open(model_path, "r") as f:
                    logger.info("reading model reference")
                    read_data = json.load(f)
                model = ModelReference(**read_data)

                # override results
                # model.results = stats["evaluation_metrics"]

                # get most up to date eval from the results.jsonl
                results = None
                if fs.exists(results_jsonl):
                    with fs.open(results_jsonl, "r") as f:
                        logger.info("loading results.jsonl")
                        results = json.loads(list(f)[-1])
                model.results = results

            else:
                model = ModelReference(
                    name,
                    data.name,
                    data.uuid,
                    hparams,
                    os.path.join("s3://", final_checkpoint) if args.remote_sync is not None else final_checkpoint,
                    version("open_lm"),
                    open_lm_args,
                    stats["evaluation_metrics"],
                    os.path.join("s3://", params_txt) if args.remote_sync is not None else params_txt,
                )
        except Exception as e:
            error = traceback.format_exc()
            print(f"Received error when creating model json:\n{error}")
            print(f"Saving model reference to failed models json.")
            model = ModelReference(
                name,
                data.name,
                data.uuid,
                hparams,
                os.path.join("s3://", final_checkpoint) if args.remote_sync is not None else final_checkpoint,
                version("open_lm"),
                open_lm_args,
                stats["evaluation_metrics"],
                os.path.join("s3://", params_txt) if args.remote_sync is not None else params_txt,
                failed=True,
                error=error,
            )
            model_path = exp_data_models_path.parent / f"failed_models/{name}.json"
            model_path.parent.mkdir(exist_ok=True, parents=True)
            remote_model_path = os.path.join(exp_root, f"failed_models/{name}.json")

        with open(model_path, "w") as f:
            logger.info(f"writing model reference to {model_path}")
            json.dump(asdict(model), f, indent=4)

        if args.remote_sync:
            print(f"Writing model reference to remote path: {remote_model_path}")
            with fs.open(remote_model_path, "w") as f:
                json.dump(asdict(model), f, indent=4)

        # clean up as needed
        if args.remote_sync is not None and args.clean_exp:
            local_exp = os.path.join(args.logs, name)

            logger.info(f"removing local experiment: {local_exp}")
            shutil.rmtree(local_exp)
