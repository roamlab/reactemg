#!/bin/bash
# Train every model variant on EMG-EPN-612 with num_classes=6.
# 6-class config keeps all gestures including {wavein, waveout, pinch}; the
# filtering in get_csv_paths automatically retains them when --num_classes is 6.
#
# Invoke from anywhere; the script `cd`s to reactemg/ so all relative paths
# (../data/EMG-EPN-612, model_checkpoints/, output/, wandb/) resolve correctly:
#   source scripts/train_all_epn_6class.sh   # from reactemg/
#   bash   scripts/train_all_epn_6class.sh   # from anywhere
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

NUM_CLASSES=6
DATASET_TAG=epn
DATASET_SELECTION=epn_only
WINDOW_SIZE=600
OFFSET=30
EPN_SUBSET=1.0
# epn_only splits a random 5% off the EPN files for validation, so --val_patient_ids
# is unused but still required by the parser. Pass a placeholder.
DUMMY_VAL=s1

# ---------- any2any (paper model) ----------
echo ""
echo "=========================================================="
echo " Training any2any_${DATASET_TAG}_${NUM_CLASSES}class"
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
echo " Training ann_${DATASET_TAG}_${NUM_CLASSES}class"
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
echo " Training lstm_${DATASET_TAG}_${NUM_CLASSES}class"
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
# ED-TCN's autoencoder uses MaxPool1d(2) twice followed by Upsample(2) twice,
# which preserves temporal length only when T_red is divisible by 4. With the
# hardcoded inner_window=150, inner_stride=25, T_red = (W - 150)/25 + 1, so
# W=600 -> T_red=19 (mismatch) and W=625 -> T_red=20 (works). Matching the
# window size that was used for prior ED-TCN runs.
ED_TCN_WINDOW=625
echo ""
echo "=========================================================="
echo " Training ed_tcn_${DATASET_TAG}_${NUM_CLASSES}class"
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
# These three args are needed so the saved args_dict matches what train-time hardcodes;
# event_classification.py rebuilds TraHGR from args_dict, so a mismatch produces a
# silently-wrong model at eval time.
echo ""
echo "=========================================================="
echo " Training trahgr_${DATASET_TAG}_${NUM_CLASSES}class"
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

# ---------- LDA baseline (sklearn fit; 1 epoch is enough since refits are deterministic) ----------
echo ""
echo "=========================================================="
echo " Training lda_${DATASET_TAG}_${NUM_CLASSES}class"
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
echo "All ${NUM_CLASSES}-class EPN models trained."
