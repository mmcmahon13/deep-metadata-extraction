#!/bin/bash

conf=$1
if [ ! -e $conf ]; then
    echo "No config file specified; Exiting."
    exit 1
fi
source $conf

debug="false"

additional_args=${@:2}

timestamp=`date +%Y-%m-%d-%H-%M-%S`
ROOT_DIR=$DEEP_META_ROOT
OUT_LOG=$ROOT_DIR/hyperparams/tune-$network-$timestamp
echo "Writing to "$OUT_LOG

if [[ "$debug" == "false" ]]; then
    mkdir -p $OUT_LOG
fi

#gpuids=( `eval $CNN_SPRED_ROOT/bin/get-free-gpus.sh | sed '1d'` )
#gpuids="1 2 3 4"
#num_gpus=${#gpuids[@]}

num_gpus=96

lrs="0.001 0.0005"
h_dropouts="0.85 0.75 0.65 0.5"
i_dropouts="0.85 0.75 0.65 0.5"
m_dropouts="1.0"
word_dropouts="0.5" # 0.85 0.65"
batch_sizes="64 32"
l2s="0.0 1e-6 1e-8"
nonlinearities="tanh" #relu sigmoid tanh"
num_filters="150" #50 100 150"
shape_dims="5"
clip_grads="1" # 5 10" # not actually used
beta2s="0.9 0.99 0.999"
epsilons="1e-4 1e-6 1e-8"
reg_drop_penalties="1e-4 1e-5"

# 6912
# 2*4*4*2*2*3*3*3*2

#viterbi_param=""
#if [[ "$viterbi" == "true" ]]; then
#    viterbi_param="--viterbi"
#fi
#
#predict_pad_param=""
#if [[ "$predict_pad" == "true" ]]; then
#    predict_pad_param="--predict_pad"
#fi

# array to hold all the commands we'll distribute
declare -a commands

for lr in ${lrs[@]}; do
    for h_dropout in ${h_dropouts[@]}; do
        for i_dropout in ${i_dropouts[@]}; do
            for m_dropout in ${m_dropouts[@]}; do
                for w_dropout in ${word_dropouts[@]}; do
                    for batch_size in ${batch_sizes[@]}; do
                        for l2 in ${l2s[@]}; do
                            for num_filter in ${num_filters[@]}; do
                                for nonlinearity in ${nonlinearities[@]}; do
                                    for shape_dim in ${shape_dims[@]}; do
                                        for clip_grad in ${clip_grads[@]}; do
                                            for beta2 in ${beta2s[@]}; do
                                                for epsilon in ${epsilons[@]}; do
                                                    for drop_penalty in ${reg_drop_penalties[@]}; do
                                                        fname_append="$lr-$beta2-$epsilon-$h_dropout-$i_dropout-$m_dropout-$w_dropout-$batch_size-$num_filter-$l2-$nonlinearity-$shape_dim-$clip_grad-$drop_penalty"

                        #                                # use CoNLL-2003 data -- load this AFTER filter_width is set
                        #                                source $CNN_SPRED_ROOT/conf/conll2003.conf

                        #                                                commands+=("CUDA_VISIBLE_DEVICES=XX python src/train.py \
                                                        commands+=("srun --gres=gpu:1 --partition=titanx-short,m40-short python src/train.py \
                                                        --model $model \
                                                        --train_dir $train_dir \
                                                        --dev_dir $dev_dir \
                                                        --max_seq_len $max_seq_len \
                                                        --embed_dim $embedding_dim \
                                                        --embeddings $embeddings \
                                                        --nonlinearity $nonlinearity \
                                                        --initialization $initialization \
                                                        --batch_size $batch_size \
                                                        --hidden_dropout $h_dropout \
                                                        --input_dropout $i_dropout \
                                                        --middle_dropout $m_dropout \
                                                        --word_dropout $w_dropout \
                                                        --lstm_dim $num_filter \
                                                        --l2 $l2 \
                                                        --lr $lr \
                                                        --beta2 $beta2 \
                                                        --epsilon $epsilon \
                                                        --until_convergence \
                                                        --shape_dim $shape_dim \
                                                        --char_dim $char_dim \
                                                        --char_tok_dim $char_tok_dim \
                                                        --model $model \
                                                        --regularize_drop_penalty $drop_penalty \
                                                        --use_geometric_feats \
                                                        --use_lexicons \
                                                        $additional_args \
                                                        &> $OUT_LOG/train-$fname_append.log")
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done

# now distribute them to the gpus
num_jobs=${#commands[@]}
jobs_per_gpu=$((num_jobs / num_gpus))
echo "Distributing $num_jobs jobs to $num_gpus gpus ($jobs_per_gpu jobs/gpu)"

j=0
for (( gpuid=0; gpuid<num_gpus; gpuid++)); do
    for (( i=0; i<jobs_per_gpu; i++ )); do
        jobid=$((j * jobs_per_gpu + i))
        comm="${commands[$jobid]}"
        comm=${comm/XX/$gpuid}
#        echo "Starting job $jobid on gpu $gpuid"
        echo ${comm}
        if [[ "$debug" == "false" ]]; then
            eval ${comm}
        fi
    done &
    j=$((j + 1))
done

