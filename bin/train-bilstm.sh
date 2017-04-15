#!/bin/bash

conf=$1
if [ ! -e $conf ]; then
    echo "No config file specified; Exiting."
    exit 1
fi
source $conf

additional_args=${@:2}

#viterbi_param=""
#if [[ "$viterbi" == "true" ]]; then
#    viterbi_param="--viterbi"
#fi
#
#predict_pad_param=""
#if [[ "$predict_pad" == "true" ]]; then
#    predict_pad_param="--predict_pad"
#fi

load_pretrained_param=""
if [[ "$pretrained_model" != "" ]]; then
    load_pretrained_param="--load_dir $pretrained_model"
fi

cmd="python src/train.py \
--model $model \
--train_dir $train_dir \
--dev_dir $dev_dir \
--model_dir $model_dir \
--max_seq_len $max_seq_len \
--embed_dim $embedding_dim \
--embeddings $embeddings \
--lstm_dim $num_filters \
--input_dropout $input_dropout \
--hidden_dropout $hidden_dropout \
--middle_dropout $middle_dropout \
--word_dropout $word_dropout \
--lr $lr \
--l2 $l2 \
--batch_size $batch_size \
--nonlinearity $nonlinearity \
--initialization $initialization \
--char_dim $char_dim \
--char_tok_dim $char_tok_dim \
--shape_dim $shape_dim \
--clip_norm $clip_grad \
--epsilon $epsilon \
--beta2 $beta2 \
--regularize_drop_penalty $drop_penalty \
$load_pretrained_param \
$additional_args"

echo ${cmd}
eval ${cmd}
