import argparse
import os
import shutil
import torch
from copy import deepcopy

from open_lm.model import create_params
from open_lm.utils.transformers.hf_config import OpenLMConfig
from open_lm.utils.transformers.hf_model import OpenLMforCausalLM, OpenLMModel

from transformers import GPTNeoXTokenizerFast, LlamaTokenizerFast, AutoTokenizer

from eval.utils import update_args_from_openlm_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments that openlm requires when we call load_model
    parser.add_argument("--checkpoint", type=str, required=True)  # TODO - make it accept s3 paths as well
    parser.add_argument(
        "--config",
        type=str,
        required=True,  # TODO - make it accept s3 paths as well
        help="a json file to be loaded by open_lm. see open_lm documentation for more info",
    )
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--force-xformers", action="store_true")
    parser.add_argument("--model", default=None, type=str)

    # args to pushing to hub
    parser.add_argument("--save_dir", type=str, default="/tmp/tmp_save_dir")
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--hf_token", type=str, required=False, default=None, help="Hugging Face API token with write permissions"
    )

    args = parser.parse_args()

    # prep save dir
    save_dir = os.path.join(args.save_dir, args.repo_id.replace("/", "_"))
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print("Done creating save_dir")

    # load tokenizer
    if "gpt-neox-20b" in args.tokenizer:
        tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
    elif "llama" in args.tokenizer:
        tokenizer = LlamaTokenizerFast.from_pretrained(args.tokenizer)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, cache_dir=args.hf_cache_dir)
    print("Done loading tokenizer")

    # load config
    args.val_data = args.resume = None
    if os.path.exists(args.config):
        if args.config.endswith(".txt"):
            update_args_from_openlm_config(args)
        else:
            assert args.config.endswith(".json"), "Model config path file must be a json file or a txt file"
    params = create_params(args)
    openlm_config = OpenLMConfig(params)
    print("Done loading config")

    # prep config to be saved (Params is not JSON serializable and so are the functions)
    params = openlm_config.params
    openlm_config.params_args_dict = deepcopy(openlm_config.params.__dict__)
    openlm_config.params = None
    del openlm_config.params_args_dict["attn_func"]
    del openlm_config.params_args_dict["norm_type"]
    for k in [
        "attn_name",
        "attn_activation",
        "attn_activation",
        "attn_seq_scalar",
        "model_norm",
        "attn_seq_scalar_alpha",
        "qk_norm",
        "positional_embedding_type",
        "ffn_type",
    ]:
        if k not in openlm_config.params_args_dict:
            openlm_config.params_args_dict[k] = getattr(args, k)
    openlm_config.params_args_dict["model"] = args.model if not os.path.isfile(args.model) else None
    # can save the config
    openlm_config.save_pretrained(save_dir)
    print("Done saving config")

    # now let's load it to see it loads fine and creates a model
    new_openlm_config = OpenLMConfig.from_pretrained(save_dir)
    open_lm = OpenLMModel(new_openlm_config)
    print("Done loading model")

    # load model
    checkpoint = torch.load(args.checkpoint)
    print("loaded checkpoint")

    # update state dict
    state_dict = checkpoint["state_dict"]
    state_dict = {x.replace("module.", "").replace("_orig_mod.", ""): y for x, y in state_dict.items()}
    print("updated state dict")

    # save the model and tokenizer
    open_lm.model.load_state_dict(state_dict)
    open_lm.config = openlm_config  # so that we save the unprocessed dict that can be serialized
    tokenizer.save_pretrained(save_dir)
    open_lm.save_pretrained(
        save_dir,
        push_to_hub=True,
        repo_id=args.repo_id,
        private=args.private,
        commit_message="updating model files",
        token=args.hf_token,
    )
    print("Done saving model")

    # make sure we can load it
    open_lm_reloaded = OpenLMforCausalLM.from_pretrained(save_dir)

    # clean up
    shutil.rmtree(save_dir)

    print("Done!")
