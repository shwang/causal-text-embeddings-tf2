#!/usr/bin/env bash
pushd src/

export BERT_BASE_DIR=../pre-trained/uncased_L-12_H-768_A-12.bak
export OUTPUT_DIR=../output/reddit_embeddings

mkdir -p ${OUTPUT_DIR}
prediction_file="${OUTPUT_DIR}/predictions.tsv"

#13,6,8 are keto, okcupid, childfree
export SUBREDDITS=13,6,8
export USE_SUB_FLAG=false
export BETA0=1.0
export BETA1=1.0
export GAMMA=1.0

tf_log_dir=../output/gender_uncertainty_tf/

# TODO: Move me up!
saved_path_root=../output/gender_uncertainty_tf/20221003_224922_future-poltergeist
saved_path_final=${saved_path_root}/trained/dragon.ckpt-1


input_reddit=../dat/reddit/proc.tf_record
input_tiny=../dat/shwang/tiny/proc.tf_record
input_small=../dat/shwang/small/proc.tf_record
input_half=../dat/shwang/half/proc.tf_record
input_full=../dat/shwang/full/proc.tf_record
input_half_tiny=../dat/shwang-half/tiny/proc.tf_record
input_half_half=../dat/shwang-half/half/proc.tf_record
input_half_full=../dat/shwang-half/full/proc.tf_record

# DATA_FILE=$input_half_half
DATA_FILE=$input_half_tiny
# DATA_FILE=$input_reddit

# Incompatible flags from TF1 run script.
  # --label_pred=true \
  # --unsupervised=true \
  # --output_dir=${OUTPUT_DIR} \
  # --num_warmup_steps=1000 \
  # --save_checkpoints_steps=5000 \
  # --keep_checkpoints=1 \
  # --num_train_steps=10000 \  # This is approximately 2 epochs.

python -m reddit.model.run_causal_bert \
  --seed=0 \
  --mode=predict_only \
  --input_files=${DATA_FILE} \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --bert_config_file=${BERT_BASE_DIR}/bert_config.json \
  --dev_splits=0 \
  --test_splits=0 \
  --max_seq_length=256 \
  --train_batch_size=16 \
  --eval_batch_size=16 \
  --learning_rate=5e-5 \
  --num_train_epochs=0 \
  --subreddits=${SUBREDDITS} \
  --beta0=${BETA0} \
  --beta1=${BETA1} \
  --gamma=${GAMMA} \
  --prediction_file=${prediction_file} \
  --model_dir=${tf_log_dir} \
  --saved_path=${saved_path_final} \
  --include_aux=false

#  --max_seq_length=512 \
# --num_train_epochs=6
# --num_train_epochs=16
