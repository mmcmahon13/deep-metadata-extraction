#!/usr/bin/env bash

# process the dev set using the maps created in the train processing
cmd="python grotoap_to_tfrecords.py \
--grotoap_dir $DATA_DIR/GROTOAP2/grotoap2/dataset/test \
--out_dir $DATA_DIR/pruned_pmc/test-30-lex/ \
--load_vocab $DATA_DIR/pruned_PMC_min_10.txt \
--load_shapes $DATA_DIR/pruned_pmc/train-30-lex/shape.txt \
--load_chars $DATA_DIR/pruned_pmc/train-30-lex/char.txt \
--load_labels $DATA_DIR/pruned_pmc/train-30-lex/label.txt \
--bilou \
--use_lexicons \
$additional_args
"
echo "processing test set:"
echo ${cmd}
eval ${cmd}