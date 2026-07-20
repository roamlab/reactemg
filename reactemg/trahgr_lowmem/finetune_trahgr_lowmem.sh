#!/bin/bash
# Low-memory TraHGR LOSO fine-tune on ROAM-EMG (28 subjects), batch size 64.
#
# Identical to scripts/finetune_trahgr_loso_batch64.sh in EVERY parameter; the
# ONLY difference is that it routes through trahgr_lowmem/run.py (lazy low-RAM
# TraHGR_Dataset, identical outputs). Nothing in the stock code is modified.
#
#   bash trahgr_lowmem/finetune_trahgr_lowmem.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"   # .../reactemg/trahgr_lowmem
cd "${SCRIPT_DIR}/.."                                            # -> reactemg/
RUN="${SCRIPT_DIR}/run.py"

# ============================================================
#   PRETRAIN CHECKPOINT — FILL IN (path relative to reactemg/)
#   Typically the output of trahgr_lowmem/train_trahgr_lowmem.sh, e.g.:
#   model_checkpoints/trahgr_pretrain_3class_<stamp>_<host>/epoch_20.pth
# ============================================================
CKPT_TRAHGR="${CKPT_TRAHGR:-TODO}"

# ---- Settings (identical to scripts/finetune_trahgr_loso_batch64.sh) ----
NUM_CLASSES=3
DATASET_SELECTION=roam_only
WINDOW_SIZE=600
OFFSET=30
EPOCHS=5
EPN_SUBSET=1.0
BATCH_SIZE=64

PATIENT_IDS=("s1" "s2" "s3" "s4" "s5" "s6" "s7" "s8" "s9" "s10" "s11" "s12" "s13" "s14" "s15" "s16" "s17" "s18" "s19" "s20" "s21" "s22" "s23" "s24" "s25" "s26" "s27" "s28")

if [ -z "$CKPT_TRAHGR" ] || [ "$CKPT_TRAHGR" = "TODO" ]; then
    echo "ERROR: set CKPT_TRAHGR to your TraHGR pretrain checkpoint first." >&2
    echo "       e.g. model_checkpoints/trahgr_pretrain_3class_<stamp>_<host>/epoch_20.pth" >&2
    exit 1
fi
if [ ! -f "$CKPT_TRAHGR" ]; then
    echo "ERROR: pretrain checkpoint not found: $CKPT_TRAHGR" >&2
    exit 1
fi
echo "Fine-tuning TraHGR (low-mem) from pretrain checkpoint: $CKPT_TRAHGR"

for pid in "${PATIENT_IDS[@]}"; do
    echo ""
    echo "=========================================================="
    echo " [low-mem] trahgr LOSO fine-tune — held-out subject = ${pid} (batch ${BATCH_SIZE})"
    echo "=========================================================="
    python3 "$RUN" \
        --model_choice trahgr \
        --num_classes $NUM_CLASSES \
        --dataset_selection $DATASET_SELECTION \
        --window_size $WINDOW_SIZE \
        --offset $OFFSET \
        --epn_subset_percentage $EPN_SUBSET \
        --val_patient_ids "$pid" \
        --epochs $EPOCHS \
        --embedding_dim 144 \
        --nhead 8 \
        --num_layers 1 \
        --batch_size $BATCH_SIZE \
        --saved_checkpoint_pth "$CKPT_TRAHGR" \
        --exp_name trahgr_LOSO_${pid}
done

echo ""
echo "All 28 TraHGR (low-mem) LOSO fine-tunes complete."
echo "  Checkpoints: model_checkpoints/trahgr_LOSO_<sN>_<stamp>_<host>/epoch_5.pth"
