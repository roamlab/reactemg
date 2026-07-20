#!/bin/bash
# Low-memory TraHGR training on all three training setups, at batch size 64.
#
# Identical to scripts/train_trahgr_batch64.sh in EVERY parameter (incl. batch 64
# and --epochs 20); the ONLY difference is that it routes through
# trahgr_lowmem/run.py, which swaps in the lazy low-RAM TraHGR_Dataset (identical
# outputs, far lower peak memory during preprocessing). Nothing in the stock code
# is modified.
#
#   bash trahgr_lowmem/train_trahgr_lowmem.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # .../reactemg/trahgr_lowmem
cd "${SCRIPT_DIR}/.."                                            # -> reactemg/
RUN="${SCRIPT_DIR}/run.py"

# ---- Common settings (identical to scripts/train_trahgr_batch64.sh) ----
WINDOW_SIZE=600
OFFSET=30
EPN_SUBSET=1.0
BATCH_SIZE=64
EPOCHS=20   # paper appendix: TraHGR EPN training/pretraining extended to 20 epochs
DUMMY_VAL=s1

# ---------- 1) EPN 3-class (epn_only) ----------
echo ""
echo "=========================================================="
echo " [low-mem] Training trahgr_epn_3class (batch ${BATCH_SIZE})"
echo "=========================================================="
python3 "$RUN" \
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
echo " [low-mem] Training trahgr_epn_6class (batch ${BATCH_SIZE})"
echo "=========================================================="
python3 "$RUN" \
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
echo " [low-mem] Pretraining trahgr_pretrain_3class (batch ${BATCH_SIZE})"
echo "=========================================================="
python3 "$RUN" \
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
echo "All TraHGR (low-mem) training runs complete. Checkpoints:"
echo "  model_checkpoints/trahgr_epn_3class_<stamp>_<host>/"
echo "  model_checkpoints/trahgr_epn_6class_<stamp>_<host>/"
echo "  model_checkpoints/trahgr_pretrain_3class_<stamp>_<host>/   (last epoch: epoch_20.pth)"
