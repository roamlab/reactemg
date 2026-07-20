#!/bin/bash
# LOSO fine-tune ONLY the TraHGR baseline on ROAM-EMG from a pretrained
# checkpoint, at batch size 64. Trains a separate model for each held-out
# subject s1..s28.
#
# Mirrors the ft_trahgr() path in scripts/finetune_all_loso.sh. The ONLY change
# vs. that script is the added `--batch_size 64`. Every other parameter
# (num_classes=3, dataset=roam_only, window=600, offset=30, epochs=5,
# embedding_dim=144, nhead=8, num_layers=1) is identical.
#
# Invoke from anywhere; the script `cd`s to reactemg/ so the relative checkpoint
# path and model_checkpoints/, output/, wandb/ folders resolve consistently:
#   bash scripts/finetune_trahgr_loso_batch64.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

# ============================================================
#   PRETRAIN CHECKPOINT — FILL IN (path relative to reactemg/)
#   Typically the output of scripts/train_trahgr_batch64.sh, e.g.:
#   model_checkpoints/trahgr_pretrain_3class_<stamp>_<host>/epoch_20.pth
# ============================================================
CKPT_TRAHGR="TODO"

# ---- Settings (identical to scripts/finetune_all_loso.sh) ----
NUM_CLASSES=3
DATASET_SELECTION=roam_only
WINDOW_SIZE=600
OFFSET=30
EPOCHS=5
# epn_subset_percentage is required by the parser but unused for roam_only.
EPN_SUBSET=1.0
BATCH_SIZE=64

PATIENT_IDS=("s1" "s2" "s3" "s4" "s5" "s6" "s7" "s8" "s9" "s10" "s11" "s12" "s13" "s14" "s15" "s16" "s17" "s18" "s19" "s20" "s21" "s22" "s23" "s24" "s25" "s26" "s27" "s28")

# ---- Require a valid pretrain checkpoint ----
if [ -z "$CKPT_TRAHGR" ] || [ "$CKPT_TRAHGR" = "TODO" ]; then
    echo "ERROR: set CKPT_TRAHGR to your TraHGR pretrain checkpoint first." >&2
    echo "       e.g. model_checkpoints/trahgr_pretrain_3class_<stamp>_<host>/epoch_20.pth" >&2
    exit 1
fi
if [ ! -f "$CKPT_TRAHGR" ]; then
    echo "ERROR: pretrain checkpoint not found: $CKPT_TRAHGR" >&2
    exit 1
fi
echo "Fine-tuning TraHGR from pretrain checkpoint: $CKPT_TRAHGR"

for pid in "${PATIENT_IDS[@]}"; do
    echo ""
    echo "=========================================================="
    echo " trahgr LOSO fine-tune — held-out subject = ${pid} (batch ${BATCH_SIZE})"
    echo "=========================================================="
    python3 main.py \
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
echo "All 28 TraHGR LOSO fine-tunes complete."
echo "  Checkpoints: model_checkpoints/trahgr_LOSO_<sN>_<stamp>_<host>/epoch_5.pth"
