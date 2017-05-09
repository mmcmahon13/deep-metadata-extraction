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

cmd="$DEEP_META_ROOT/bin/train-bilstm-basic-char.sh \
$conf \
--evaluate_only \
--train_eval \
--load_dir bilstm-char-frontend \
--dev_dir $dev_dir \
$additional_args"

echo ${cmd}
eval ${cmd}