#!/bin/bash
# Pretrain every model variant on the combined public + EPN dataset
# ("pub_with_epn") at 3 classes, in preparation for downstream LOSO
# fine-tuning on ROAM-EMG (see scripts/finetune_runner.sh).
#
# Differences vs. scripts/train_all_epn_3class.sh:
#   * --dataset_selection pub_with_epn  (vs. epn_only):
#       - adds SS-STM_for_Myo_filtered, Mangalore_University, ROSHAMBO
#       - EPN auto-filtered to "open"/"fist" files only (matches the
#         3-class gesture set common to the public datasets)
#       - val = random 5% of the combined pool (val_patient_ids unused)
#   * exp_name suffix changes from `_epn_` to `_pretrain_` so
#     pretrain checkpoints don't collide with EPN-only ones.
#   * Everything else (window_size, offset, model-specific knobs) is
#     identical to train_all_epn_3class.sh.
#
# Invoke from anywhere; the script `cd`s to reactemg/:
#   source scripts/pretrain_all_3class.sh   # from reactemg/
#   bash   scripts/pretrain_all_3class.sh   # from anywhere
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

NUM_CLASSES=3
DATASET_TAG=pretrain
DATASET_SELECTION=pub_with_epn
WINDOW_SIZE=600
OFFSET=30
EPN_SUBSET=1.0
# pub_with_epn ignores val_patient_ids (random 5% split); arg still required by parser.
DUMMY_VAL=s1

# ---------- any2any (paper pretraining target) ----------
echo ""
echo "=========================================================="
echo " Pretraining any2any_${DATASET_TAG}_${NUM_CLASSES}class"
echo "=========================================================="
python3 main.py \
  --model_choice any2any \
  --num_classes $NUM_CLASSES \
  --dataset_selection $DATASET_SELECTION \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --embedding_method linear_projection \
  --use_input_layernorm \
  --share_pe \
  --use_warmup_and_decay \
  --task_selection 0 1 2 \
  --exp_name any2any_${DATASET_TAG}_${NUM_CLASSES}class

# ---------- ANN baseline ----------
echo ""
echo "=========================================================="
echo " Pretraining ann_${DATASET_TAG}_${NUM_CLASSES}class"
echo "=========================================================="
python3 main.py \
  --model_choice ann \
  --num_classes $NUM_CLASSES \
  --dataset_selection $DATASET_SELECTION \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --exp_name ann_${DATASET_TAG}_${NUM_CLASSES}class

# ---------- LSTM baseline ----------
echo ""
echo "=========================================================="
echo " Pretraining lstm_${DATASET_TAG}_${NUM_CLASSES}class"
echo "=========================================================="
python3 main.py \
  --model_choice lstm \
  --num_classes $NUM_CLASSES \
  --dataset_selection $DATASET_SELECTION \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --exp_name lstm_${DATASET_TAG}_${NUM_CLASSES}class

# ---------- ED-TCN baseline ----------
# ED-TCN's MaxPool/Upsample autoencoder requires T_red divisible by 4. With
# the hardcoded inner_window=150, inner_stride=25, T_red = (W - 150)/25 + 1,
# so W=600 -> 19 (mismatch) and W=625 -> 20 (works). Same window_size override
# as the EPN training script.
ED_TCN_WINDOW=625
echo ""
echo "=========================================================="
echo " Pretraining ed_tcn_${DATASET_TAG}_${NUM_CLASSES}class"
echo " (window_size=${ED_TCN_WINDOW} so ED-TCN's T_red is divisible by 4)"
echo "=========================================================="
python3 main.py \
  --model_choice ed_tcn \
  --num_classes $NUM_CLASSES \
  --dataset_selection $DATASET_SELECTION \
  --window_size $ED_TCN_WINDOW \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --exp_name ed_tcn_${DATASET_TAG}_${NUM_CLASSES}class

# ---------- TraHGR baseline (paper "TraHGR-Huge" config: D=144, h=8, L=1) ----------
# These three args are needed so the saved args_dict matches what train-time
# hardcodes; event_classification.py rebuilds TraHGR from args_dict, so a
# mismatch produces a silently-wrong model at eval time.
echo ""
echo "=========================================================="
echo " Pretraining trahgr_${DATASET_TAG}_${NUM_CLASSES}class"
echo "=========================================================="
python3 main.py \
  --model_choice trahgr \
  --num_classes $NUM_CLASSES \
  --dataset_selection $DATASET_SELECTION \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --embedding_dim 144 \
  --nhead 8 \
  --num_layers 1 \
  --exp_name trahgr_${DATASET_TAG}_${NUM_CLASSES}class

# ---------- LDA baseline ----------
# Note: LDA "pretraining" produces a fitted sklearn estimator on pub_with_epn,
# but downstream fine-tuning would call model.fit_numpy() again on roam_only
# and overwrite that fit entirely (sklearn LDA has no warm-start). Included
# for completeness; useful only if you want a baseline LDA trained on the
# broader pool to compare against an LDA trained on ROAM alone.
echo ""
echo "=========================================================="
echo " Pretraining lda_${DATASET_TAG}_${NUM_CLASSES}class"
echo "=========================================================="
python3 main.py \
  --model_choice lda \
  --num_classes $NUM_CLASSES \
  --dataset_selection $DATASET_SELECTION \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --epochs 1 \
  --exp_name lda_${DATASET_TAG}_${NUM_CLASSES}class

echo ""
echo "All pretrain models trained on ${DATASET_SELECTION} (${NUM_CLASSES}-class)."
echo "Pretrain checkpoints are under model_checkpoints/<model>_${DATASET_TAG}_${NUM_CLASSES}class_<stamp>_<host>/."
echo "For downstream LOSO fine-tuning on ROAM-EMG:"
echo "  edit scripts/finetune_runner.sh's saved_checkpoint_pth to the any2any pretrain checkpoint,"
echo "  then run: source scripts/finetune_runner.sh"
