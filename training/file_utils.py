import json
from pathlib import Path
import hashlib
import warnings
import urllib.request
import os
import logging
import re
import subprocess
import time
import fsspec
import torch
import yaml

import multiprocessing as mp

from dataclasses import asdict
from importlib.metadata import version
from tqdm import tqdm
from training.model_reference import ModelReference


tok_mult_paths = [
    "training/eval_data/val_tok_mult/openlm/shard_00000000.tar",
    "training/eval_data/c4_val/shard-{0000000..0000010}.tar",
    # "training/eval_data/paloma_val/{00000001..00000004}.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_000.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_010.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_020.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_030.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_040.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_050.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_060.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_070.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_080.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_090.tar",
    # "training/eval_data/val_tok_mult/en-de_mix_shards/val_en-de_100.tar",
    "training/eval_data/val_tok_mult/paloma_4chan_meta_sep/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_c4_100_domains/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_c4_en/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_dolma-v1_5/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_dolma_100_programing_languages/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_dolma_100_subreddits/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_falcon-refinedweb/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_gab/00000001.tar",
    # "training/eval_data/val_tok_mult/paloma_m2d2_s2orc_unsplit/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_m2d2_s2orc_unsplit_dedup/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_m2d2_wikipedia_unsplit/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_manosphere_meta_sep/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_mc4/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_ptb/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_redpajama/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_twitterAAE_HELM_fixed/00000001.tar",
    "training/eval_data/val_tok_mult/paloma_wikitext_103/00000001.tar",
]

# for safety, don't be sorry, be safe
DOWNSTREAM_SHARD_HASHES = {
    "agi_eval_lsat_ar": "f74c0426e8bd08e1e9bcf758ccc1353ca45415a672fa1c763294c3436fee1037",
    "agi_eval_sat_math": "50c64867548c1e76a65051652fca2a2e30e6cbfe6b55eafe00b1c463536565f4",
    "aqua": "7d6823fad372f1e42cd7b43f344d7d1947d0c97a3f9a0f574b22160f107dadca",
    "bigbench_cs_algorithms": "64ed919e3a3ea0ea526af105c173ddd3ba1267c2f6bda54755a2198850bc645a",
    "bigbench_dyck_languages": "77f5eacf426a26b63411e1b33e7e1d4429cf6f6433c4e4f7cf0dcd2aac477520",
    "bigbench_elementary_math_qa": "0e45b1bb172f0b1b0062f0dfd59ab554a43dc7cf876c10f2276387f0809c1b60",
    "bigbench_logical_deduction": "81deb91ab888ef542ce28ac2d63a7fa1a258b981cf0dc6f4fff44fe335ba89e3",
    "bigbench_operators": "79fe39b2d1027394ac0122bb3862e0db7b7f3bc9e267b996077ce2f9319f0e8f",
    "bigbench_repeat_copy_logic": "72fb7fb2a14c9e32103ef9dd62cd8f9c59fb407c9ee69b3b663782b84e661952",
    "gsm8k": "4fea4cbd4aca9c3bcc222134d907e4f94d5ef624c0d69616e7967e169b41decb",
    "logi_qa": "2bec953987b78285e0be404636bf42ccd5d07fee11dc8e6617139826d7895894",
    "math_qa": "fb9849cb1730f671f9dc96951d1aff6d66a7e2bcff35e4e409904d4b401695f1",
    "simple_arithmetic_nospaces": "fb476233cd5cdc6a2306653810ec966181486a28ff54067f5d61aced70f98235",
    "simple_arithmetic_withspaces": "63fb46e6374cb191db78a9dbdfd154a9e857534307aaca909e795a819968ef74",
    "svamp": "8ac6a4a1033cd7d4cee52d3374253f875a7d38e188aeeabadd8cf1a577e05552",
    "agi_eval_lsat_lr": "40fc30f3eb4e9ab34102cd761b8c6fcaa12afd4b0be9d30a1ca8f5c0e87295ae",
    "agi_eval_lsat_rc": "345f8d3c5ab4f5db2e9c41d669db0f4167f4fd74f6d30d3d02d156648d283f3f",
    "agi_eval_sat_en": "0ec66638aa2760a1f521cc1723a32e5a17772cae2dc7634fccf7bf1ad28a56c6",
    "bigbench_understanding_fables": "9bdaf1433c0ab977802b93ce941ad98f3345321faf81b31d782032b3177f8335",
    "boolq": "8a2b46e5adccd171874c3070cf3843251a038e078eaa7d55a5f29afda651b076",
    "coqa": "2a25d60a50aecee2877f5e7f512ce751f57e7993ff027d3ee18fe00523fe0c36",
    "pubmed_qa_labeled": "71e1b98d340c39a66d3373f5f81b56a7af73f3d6c6027572ca13d8e58ea4bddd",
    "squad": "b03b19345e109dbbace401cd660b50850b90839b4aa56caa3d8975796780479b",
    "human_eval": "7dc32b1a52249c677208a1626b72478c312d4eabd364d5336a9b5581626c4271",
    "human_eval-0.25": "67723a17c6f6e6a75537b23db15c7efbfe3a90e3b32719ca6a790f51edeb890d",
    "human_eval-0.5": "75943f7649a6914e4e3a1aca7d8d07230484e5b10c5d13da499a28c62fd2e105",
    "human_eval-0.75": "f32c8cb414162bacb2f693635615856aff9ce10baa4695bddcf01d257173666a",
    "human_eval_return_complex": "dc7043d78523ac91f81d989e09201587a95770b8f15fa37dd639caa54b2c9a8e",
    "human_eval_return_simple": "f3b3980f268358de0f2b4096c46a74df8df7eec15027568d44f0541eab4d9765",
    "processed_human_eval_cpp": "cc70cd2a24efa63d1e29a05993fee1231c63a966cfcdc43fcbda4c70df53cc78",
    "processed_human_eval_js": "a1e9874570525b12e6a267e7b7656170adf59bfe64bfdcebf0e1cb3d6c8299ae",
    "bigbench_conceptual_combinations": "b17fba80117abd9b9af396270fe5691eafafa08853dbc246a5769cd5b662e29c",
    "bigbench_conlang_translation": "3ef864110ccc33bc63d8736c26c798afe3ac82507a1bbce62192cb41f3b7870a",
    "bigbench_language_identification": "84a6dac7cf0ba60737058617feaabf4d4c4bc3bb98fd668f8c3df9905b769d69",
    "hellaswag": "4c5addf39903571c16dc6e8261eb8f05e92cf8b0eb58490b61e84938844d8ba4",
    "lambada_openai": "a8699b4fc07a721e7d9cd58cba0060aa6e8b63c60528485f555c7cb6bd38fa8c",
    "winograd_wsc": "d948bfbdc147d3d494c6e18939405cec48600c28d2ea7f29af45a553a01dbb71",
    "winogrande": "da27270a585f96c26b7563761018c6770172a18d3e4408d6e2048c6b341d770d",
    "bigbench_novel_concepts": "e5e2ac306b86587ae6d78c178bed49e7081c5ed29fdf442bf70cacc0f11435eb",
    "bigbench_strange_stories": "7f46194b3793853f7c98b3d1865a870872c5f0c25c587f97ddca0fffeb1ae3a6",
    "bigbench_strategy_qa": "68a39c750e8905ac6eecb88b5f1e072b3124edbe01d6e11d3944890ed2819938",
    "commonsense_qa": "91e756431c9047d81e9d5cfc14670dee42581cfa1b089235ba269123b258875b",
    "copa": "a5e6055c55eef4451f4990df2d7d20c2a0f1f98cea2b9609e8d8817e0e85e884",
    "openbook_qa": "f725fbc769f7d630038fe59a71a373762ff4e8390855c47ea6077a62cdb0181b",
    "piqa": "0b40a9b951aade9296ee89ba1d8b2fd7cf06edffd2a7be7a11098d78449f5259",
    "siqa": "f4857ed4b452cdae80651f69e22af4f0377af4f8a0b6700e86e4166b97018587",
    "bbq": "63241d0baef9e38c0adb4475293075b970dd72d4fd3dd1159e989781d1467167",
    "enterprise_pii_classification": "c09f8666baffc9f955743368cb4404482b8291157f6ec5c379b03a3d8351ccef",
    "winogender_mc_female": "3ad31a1e5a0021997ecac93f0fed447f2bd397a2c0c776d8fe2090a83c8b22c3",
    "winogender_mc_male": "98624b71df4107f95bc4a3d82e00435bb3f2690a16e994ded5bfb9cabbc2eca5",
    "arc_challenge": "2047faa319cd5939b582ff5b27e8d18633cc6fca972abc4e64fee36ae2750d5f",
    "arc_easy": "f7264add0515608cb0f21404b10d9c8abbd61125d742a672535f2e84db303556",
    "bigbench_misconceptions": "433700ec50bc62200948d6b4b9d902a545c60f0a7c669f14b2263c245760557e",
    "bigbench_qa_wikidata": "2d65962b62a5c8fed3a6e143f209646a32271686a00765d2b8fe29dce385357f",
    "jeopardy_all": "f937b9bd32460994bd391ae3e4dea83cdd4078b01ee1ed906cf32a88a942c8e9",
    "mmlu": "9c96c8418d9e43d35dea841b2fa1d57162e1cb0a3dc693314ba1673e78c0ea57",
    "triviaqa_sm_sub": "8e962e6cd4ab9686c86ff2bc95672221b779e5a3200d49486e255c2c5ba359fe",
}


def get_downstream_task_name(task):
    return ".".join(task["dataset_uri"].split("/")[-1].split(".")[:-1])


def load_ppl_yaml(size="heavy"):
    task_yml = os.path.join(
        Path(__file__).parent.parent.absolute(),
        "eval",
        f"{size}_ppl.yaml",
    )

    tasks = None
    with open(task_yml, "r") as stream:
        try:
            tasks = yaml.safe_load(stream)["icl_tasks"]

        except yaml.YAMLError as exc:
            print(exc)

    return {get_downstream_task_name(task): task for task in tasks}


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def download_val_data(name, root=Path(__file__).parent / f"eval_data/", skip_download=False):
    # modified from oai _download clip function

    if root is None:
        raise RuntimeError(f"{root} must not be None")

    cloud_checkpoints = {
        "open_lm_val": {
            "passable_name": os.path.join(root, name, "shard_00000000.tar"),
            "downloads": [
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/open_lm_example_data/resolve/main/validation_data/shard_00000000.tar",
                    "sha256": "68716cb8a326c8a2d15fab0bc77a6146139a9b20ba8b0003ddc6e053242a3d50",
                },
            ],
        },
        "c4_val": {
            "passable_name": os.path.join(root, name, "shard-{0000000..0000010}.tar"),
            "downloads": [
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000000.tar",
                    "sha256": "a2cd985d4f97a0a04fa28e7cfceca87365ec7b83ea6b27cc8962f707a8aeda01",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000001.tar",
                    "sha256": "29d3a0b921fcabc5c2e628a7dc16de34780c7176b46d86b7034042c13f5296e8",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000002.tar",
                    "sha256": "19e12214dabe7a8598df2f090c16d314938bd3f447e30ae123aab4ca957786ed",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000003.tar",
                    "sha256": "684697bd7c23de4b1f69de85f8975293f03503de0caf927c21eba22d0d4d23e6",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000004.tar",
                    "sha256": "2dec8cd89bc69ba18e09d14fadb528e991c842e3c97f511215ca304cbb46d086",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000005.tar",
                    "sha256": "d741f4a0621e6989014eaa88f87b7256f2f4efaad1bcc0468c8917ea50967740",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000006.tar",
                    "sha256": "66c41791b0843b2bf5e297d19dd96929c099de494fb1b750377c2fdcf9328c94",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000007.tar",
                    "sha256": "59461734c71c59c18aeebffe3bee350526ee2506b446c129190169a45633fea5",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000008.tar",
                    "sha256": "d5a9fea2897a40b7c718008667009bcc3117d5350505e7e5e0702e29a2487edf",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000009.tar",
                    "sha256": "36151788110b20486fcba2a3c40e6a63f81a81cf18efdd01a927df13fb9cd74e",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/c4_validation/resolve/main/shard-0000010.tar",
                    "sha256": "834bc6d9f8f1413d9b67bc9e2188ca72da02fa02a4c7a2fd9980ea21517377e4",
                },
            ],
        },
        "paloma_val": {
            "passable_name": os.path.join(root, name, "{00000001..00000004}.tar"),
            "downloads": [
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/paloma_validation/resolve/main/00000001.tar",
                    "sha256": "27486aa41aba5471992a0d69414bc3c2d90c9960f4fec1c1d8d3f567eb799177",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/paloma_validation/resolve/main/00000002.tar",
                    "sha256": "f7a2331f83531b00fb4b64ef62908895350d8b08490ee355756f0529453eba3d",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/paloma_validation/resolve/main/00000003.tar",
                    "sha256": "62f5d20d3bac68582c4a1b22d128c1319cddb083e64ddb4104bf1c37c8fda6a8",
                },
                {
                    "url": "https://huggingface.co/datasets/mlfoundations/paloma_validation/resolve/main/00000004.tar",
                    "sha256": "8cf6d619054ca783b1559801f0ef1c9bd64b3a2fe0eab5fb6641fdeab248c3f2",
                },
            ],
        },
    }

    if name in DOWNSTREAM_SHARD_HASHES:
        # case where request a special downstream shard for eval, populate accordingly

        tasks = load_ppl_yaml()
        category = tasks[name]["dataset_uri"].split("/")[1]

        cloud_checkpoints[name] = {
            "passable_name": os.path.join(root, name, "shard-0000000.tar"),
            "downloads": [
                {
                    "url": f"https://huggingface.co/datasets/mlfoundations/downstream_validation/resolve/main/{category}/{name}/shard-0000000.tar",
                    "sha256": DOWNSTREAM_SHARD_HASHES[name],
                },
            ],
        }

    if name not in cloud_checkpoints:
        raise ValueError(
            f"unsupported cloud checkpoint: {name}. currently we only support: {list(cloud_checkpoints.keys())}"
        )

    os.makedirs(os.path.join(root, name), exist_ok=True)

    for payload in cloud_checkpoints[name]["downloads"]:
        if skip_download:
            continue

        expected_sha256 = payload["sha256"]
        download_target = os.path.join(root, name, payload["url"].split("/")[-1])
        url = payload["url"]

        if os.path.exists(download_target) and not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")

        if os.path.isfile(download_target):
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
                )

        with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
            with tqdm(
                total=int(source.info().get("Content-Length")),
                ncols=80,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as loop:
                while True:
                    buffer = source.read(8192)
                    if not buffer:
                        break

                    output.write(buffer)
                    loop.update(len(buffer))

        # if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        #     raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return cloud_checkpoints[name]["passable_name"]


def setup_logger(name=__name__):
    logger = logging.getLogger(name)

    # Set the logging level
    logger.setLevel(logging.DEBUG)  # For example, set level to DEBUG

    # Create a StreamHandler for stdout
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.INFO)  # Optionally set a different level for stdout

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(formatter)

    # Add the stdout handler to the logger
    logger.addHandler(stdout_handler)

    return logger


def maybe_save_model_json(partial_model_dir, exp_root, name, data, hparams, open_lm_args, latest_ckpt):

    logging.info("Checking for partial model.")
    fs, path = fsspec.core.url_to_fs(exp_root)
    if exp_root.startswith("s3://"):
        result = subprocess.run(
            ["aws", "s3", "ls", os.path.join(exp_root, "checkpoints"), "--recursive"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        files = [l.strip().split(" ")[-1] for l in result.stdout.decode().splitlines()]
        stats = [os.path.join(exp_root, "checkpoints", Path(f).name) for f in files if "stats" in Path(f).name]
    else:
        stats_glob = os.path.join(path, "checkpoints", "stats_*.pt")
        stats = fs.glob(stats_glob)
    stats = sorted(stats, key=natural_key)
    if not stats:
        logging.info("No stats found, sleeping for now.")
        return latest_ckpt
    latest_stats = stats[-1]
    latest_stats_idx = int(Path(latest_stats).with_suffix("").name.split("_")[-1])
    if latest_stats_idx <= latest_ckpt:
        logging.info("No newer model found, sleeping for now.")
        return latest_ckpt
    else:
        logging.info(f"Model with epoch {latest_stats_idx} found, saving.")
        with fs.open(latest_stats, "rb") as f:
            stats = torch.load(f)

        latest_checkpoint = latest_stats.replace("stats", "epoch")
        params_txt = os.path.join(exp_root, "params.txt")

        model_ref = ModelReference(
            f"{name}_partial_{latest_stats_idx}",
            data.name,
            data.uuid,
            hparams,
            latest_checkpoint,
            version("open_lm"),
            open_lm_args,
            stats["evaluation_metrics"],
            params_txt,
        )

        partial_model_path = os.path.join(partial_model_dir, f"{name}_partial_epoch_{latest_stats_idx}.json")
        with open(partial_model_path, "w") as f:
            json.dump(asdict(model_ref), f, indent=4)

        return latest_stats_idx


def keep_running_model_json(partial_model_dir, exp_root, name, data, hparams, open_lm_args, wait_secs):
    logging.info("Starting partial model saving process.")
    latest_ckpt = 0
    while True:
        time.sleep(wait_secs)
        latest_ckpt = maybe_save_model_json(partial_model_dir, exp_root, name, data, hparams, open_lm_args, latest_ckpt)


def start_partial_model_process(partial_model_dir, exp_root, name, data, hparams, open_lm_args, wait_secs=600):
    p = mp.Process(
        target=keep_running_model_json,
        args=(partial_model_dir, exp_root, name, data, hparams, open_lm_args, wait_secs),
    )
    return p


def terminate_partial_model_process(p: mp.Process):
    if p is not None and p.is_alive():
        logging.info(f"Terminating remote sync process.")
        p.terminate()
