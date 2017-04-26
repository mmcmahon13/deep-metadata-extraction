#!/bin/bash

conf=$1
if [ ! -e $conf ]; then
    echo "No config file specified; Exiting."
    exit 1
fi
source $conf

additional_args=${@:2}

viterbi_param=""
if [[ "$viterbi" == "true" ]]; then
    viterbi_param="--viterbi"
fi

layers2_param=""
if [[ "$layers2" != "" ]]; then
    layers2_param="--layers2 \"$layers2\""
fi

if [[ "$2" == "test" ]]; then
    dev_dir=$test_dir
    additional_args=${@:3}
fi

cmd="$CNN_SPRED_ROOT/bin/train-lstm.sh \
$conf \
--evaluate_only \
--train_eval \
--load_dir $model_dir-frontend \
--dev_dir $dev_dir \
$additional_args"

echo ${cmd}
eval ${cmd}