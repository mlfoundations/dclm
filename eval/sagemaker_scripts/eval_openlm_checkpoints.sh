set -eux

python launch_on_sagemaker.py --entry-point eval/eval_openlm_ckpt.py --cfg-path ${1} --model_uri ${2} --user ${3:-${USER}} --instance-type ${4:-"p4"} --profile ${5:-"poweruser"}  --build ${@:6}
