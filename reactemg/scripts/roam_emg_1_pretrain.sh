#!/bin/bash
# ROAM-EMG pipeline — STEP 1 of 3: pretrain.
#
# Pretrains the any2any (ReactEMG) model on the combined public + EPN dataset
# ("pub_with_epn") at 3 classes. This produces the checkpoint that step 2
# (scripts/roam_emg_2_finetune.sh) fine-tunes onto ROAM-EMG via LOSO.
#
# This is the SINGLE-MODEL pipeline for reproducing the main method. To also
# pretrain every baseline (ann, lstm, ed_tcn, trahgr, lda), use the broader
# scripts/pretrain_all_3class.sh instead.
#
# Notes:
#   * pub_with_epn validates on a random 5% of the combined pool, so
#     --val_patient_ids is unused here but still required by the parser
#     (s1 is just a placeholder).
#   * Pretraining defaults to --epochs 12; any2any checkpoints are 0-indexed,
#     so the last one is epoch_11.pth (what step 2 picks up by default).
#
# Invoke from anywhere; the script `cd`s to reactemg/ so all relative paths
# (../data, model_checkpoints/, output/, wandb/) resolve correctly:
#   bash scripts/roam_emg_1_pretrain.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

echo ""
echo "=========================================================="
echo " ROAM-EMG step 1/3: pretrain any2any on pub_with_epn (3-class)"
echo "=========================================================="
python3 main.py \
  --model_choice any2any \
  --num_classes 3 \
  --dataset_selection pub_with_epn \
  --window_size 600 \
  --offset 30 \
  --epn_subset_percentage 1.0 \
  --val_patient_ids s1 \
  --embedding_method linear_projection \
  --use_input_layernorm \
  --share_pe \
  --use_warmup_and_decay \
  --task_selection 0 1 2 \
  --exp_name any2any_pretrain_3class

echo ""
echo "Pretraining complete."
echo "  Checkpoints: model_checkpoints/any2any_pretrain_3class_<stamp>_<host>/epoch_<N>.pth"
echo "  Next:        bash scripts/roam_emg_2_finetune.sh"
echo "               (auto-uses the latest pretrain run's epoch_11.pth)"
