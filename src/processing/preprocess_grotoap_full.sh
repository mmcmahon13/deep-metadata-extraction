#!/usr/bin/env bash

#!/bin/bash

export DATA_DIR="/iesl/canvas/mmcmahon/data/"

# python grotoap_to_tfrecords.py --out_dir $DATA_DIR/pruned_pmc/train-30-lex-xlabels --load_vocab
# $DATA_DIR/pruned_PMC_min_10.txt --grotoap_dir /iesl/canvas/mmcmahon/data/GROTOAP2/grotoap2/dataset/train --bilou --use_lexicons

additional_args=${@:1}

# process the train set
cmd="python grotoap_to_tfrecords.py \
--grotoap_dir $DATA_DIR/GROTOAP2/grotoap2/dataset/train \
--out_dir $DATA_DIR/pruned_pmc/train-full/ \
--load_vocab $DATA_DIR/pruned_PMC_min_10.txt \
--bilou \
--use_lexicons \
$additional_args
"
echo "processing larger train set"
echo ${cmd}
eval ${cmd}

# process the dev set using the maps created in the train processing
cmd="python grotoap_to_tfrecords.py \
--grotoap_dir $DATA_DIR/GROTOAP2/grotoap2/dataset/test/ \
--out_dir $DATA_DIR/pruned_pmc/test-full/ \
--load_vocab $DATA_DIR/pruned_PMC_min_10.txt \
--load_shapes $DATA_DIR/pruned_pmc/train-full/shape.txt \
--load_chars $DATA_DIR/pruned_pmc/train-full/char.txt \
--load_labels $DATA_DIR/pruned_pmc/train-full/label.txt \
--bilou \
--use_lexicons \
$additional_args
"
echo "processing test set:"
echo ${cmd}
eval ${cmd}
