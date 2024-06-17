set -eux

python launch_on_sagemaker.py --entry-point eval/eval_openlm_ckpt.py --cfg-path ${1} --user ${2:-${USER}} --instance-type ${3:-"p4"} --profile ${4:-"default"}  --update