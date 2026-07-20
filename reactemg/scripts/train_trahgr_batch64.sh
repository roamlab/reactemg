#!/bin/bash
# Train ONLY the TraHGR baseline on all three training setups, at batch size 64.
#
# Mirrors the TraHGR invocations in:
#   scripts/train_all_epn_3class.sh   -> EPN 3-class (epn_only)
#   scripts/train_all_epn_6class.sh   -> EPN 6-class (epn_only)
#   scripts/pretrain_all_3class.sh    -> pretrain on pub_with_epn (3-class)
#
# Changes vs. those scripts: `--batch_size 64` and `--epochs 20`. The 20 epochs
# matches the paper's appendix, which extended TraHGR EPN training/pretraining to
# 20 epochs to converge (main.py's default of 12 under-trains TraHGR). All other
# TraHGR parameters are identical.
#
# Invoke from anywhere; the script `cd`s to reactemg/ so all relative paths
# (../data, model_checkpoints/, output/, wandb/) resolve correctly:
#   bash scripts/train_trahgr_batch64.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

# ---- Common settings (identical to the original scripts) ----
WINDOW_SIZE=600
OFFSET=30
EPN_SUBSET=1.0
BATCH_SIZE=64
EPOCHS=20   # paper appendix: TraHGR EPN training/pretraining extended to 20 epochs
# epn_only / pub_with_epn use a random 5% validation split, so --val_patient_ids
# is unused but still required by the parser. Pass a placeholder.
DUMMY_VAL=s1

# ---------- 1) EPN 3-class (epn_only) ----------
echo ""
echo "=========================================================="
echo " Training trahgr_epn_3class (batch ${BATCH_SIZE})"
echo "=========================================================="
python3 main.py \
  --model_choice trahgr \
  --num_classes 3 \
  --dataset_selection epn_only \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --embedding_dim 144 \
  --nhead 8 \
  --num_layers 1 \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --exp_name trahgr_epn_3class

# ---------- 2) EPN 6-class (epn_only) ----------
echo ""
echo "=========================================================="
echo " Training trahgr_epn_6class (batch ${BATCH_SIZE})"
echo "=========================================================="
python3 main.py \
  --model_choice trahgr \
  --num_classes 6 \
  --dataset_selection epn_only \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --embedding_dim 144 \
  --nhead 8 \
  --num_layers 1 \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --exp_name trahgr_epn_6class

# ---------- 3) Pretrain on pub_with_epn (3-class) ----------
echo ""
echo "=========================================================="
echo " Pretraining trahgr_pretrain_3class (batch ${BATCH_SIZE})"
echo "=========================================================="
python3 main.py \
  --model_choice trahgr \
  --num_classes 3 \
  --dataset_selection pub_with_epn \
  --window_size $WINDOW_SIZE \
  --offset $OFFSET \
  --epn_subset_percentage $EPN_SUBSET \
  --val_patient_ids $DUMMY_VAL \
  --embedding_dim 144 \
  --nhead 8 \
  --num_layers 1 \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --exp_name trahgr_pretrain_3class

echo ""
echo "All TraHGR training runs complete. Checkpoints:"
echo "  model_checkpoints/trahgr_epn_3class_<stamp>_<host>/"
echo "  model_checkpoints/trahgr_epn_6class_<stamp>_<host>/"
echo "  model_checkpoints/trahgr_pretrain_3class_<stamp>_<host>/   (last epoch: epoch_20.pth)"
echo ""
echo "Next: fine-tune on ROAM by pointing scripts/finetune_trahgr_loso_batch64.sh"
echo "at the pretrain checkpoint (epoch_20.pth)."
