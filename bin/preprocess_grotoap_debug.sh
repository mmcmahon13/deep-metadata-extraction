#!/bin/bash

conf=$1
if [ ! -e $conf ]; then
    echo "No config file specified; Exiting."
    exit 1
fi
source $conf

additional_args=${@:2}

output_dir="$DEEP_META_ROOT/data/$data_name-$embeddings_name"
vocab_param="--vocab $embeddings"
labels_param=""
char_param=""
shape_param=""

lower_param=""
if [[ "$lowercase" == "true" ]]; then
    lower_param="--lower"
fi

update_vocab_param=""
if [[ "$update_vocab_file" != "" ]]; then
    update_vocab_param="--update_vocab $update_vocab_file"
fi

echo "Writing output to $output_dir"

cmd="python src/processing/grotoap_to_tfrecords.py \
    --grotoap_dir $this_data_file \
    --out_dir $this_output_dir \
    --load_vocab $filter_width"