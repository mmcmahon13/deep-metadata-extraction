#!/bin/bash

export DATA_DIR="/iesl/canvas/mmcmahon/data/"

# python grotoap_to_tfrecords.py --out_dir $DATA_DIR/pruned_pmc/train-30-lex-xlabels --load_vocab
# $DATA_DIR/pruned_PMC_min_10.txt --grotoap_dir /iesl/canvas/mmcmahon/data/GROTOAP2/grotoap2/dataset/train --bilou --use_lexicons

additional_args=${@:1}

# process the train set
cmd="python grotoap_to_tfrecords.py \
--grotoap_dir $DATA_DIR/GROTOAP2/grotoap2/dataset/train \
--out_dir $DATA_DIR/pruned_pmc/train-30-lex/ \
--load_vocab $DATA_DIR/pruned_PMC_min_10.txt \
--bilou \
--use_lexicons \
$additional_args
"
echo "processing train set:"
echo ${cmd}
eval ${cmd}

# process the dev set using the maps created in the train processing
cmd="python grotoap_to_tfrecords.py \
--grotoap_dir $DATA_DIR/GROTOAP2/grotoap2/dataset/dev \
--out_dir $DATA_DIR/pruned_pmc/dev-30-lex/ \
--load_vocab $DATA_DIR/pruned_PMC_min_10.txt \
--load_shapes $DATA_DIR/pruned_pmc/train-30-lex/shape.txt \
--load_chars $DATA_DIR/pruned_pmc/train-30-lex/char.txt \
--load_labels $DATA_DIR/pruned_pmc/train-30-lex/label.txt \
--bilou \
--use_lexicons \
$additional_args
"
echo "processing dev set:"
echo ${cmd}
eval ${cmd}
