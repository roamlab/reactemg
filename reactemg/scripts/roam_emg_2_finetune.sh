#!/bin/bash
# ROAM-EMG pipeline — STEP 2 of 3: LOSO fine-tune.
#
# Fine-tunes the pretrained any2any model on ROAM-EMG under a
# leave-one-subject-out (LOSO) protocol: one model per held-out subject
# s1..s28 (28 models total).
#
# By default this auto-discovers the most recent any2any pretrain run from
# step 1 (scripts/roam_emg_1_pretrain.sh) and fine-tunes from its last epoch
# (epoch_11.pth). Override the starting checkpoint by exporting PRETRAIN_CKPT,
# e.g. to use a different/best epoch:
#   PRETRAIN_CKPT=model_checkpoints/any2any_pretrain_3class_<stamp>_<host>/epoch_11.pth \
#     bash scripts/roam_emg_2_finetune.sh
#
# This is the SINGLE-MODEL pipeline for reproducing the main method. To fine-tune
# every baseline as well, use the broader scripts/finetune_all_loso.sh instead.
#
# Invoke from anywhere; the script `cd`s to reactemg/:
#   bash scripts/roam_emg_2_finetune.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

NUM_CLASSES=3
DATASET_SELECTION=roam_only
WINDOW_SIZE=600
OFFSET=30
EPOCHS=5
# epn_subset_percentage is required by the parser but unused for roam_only.
EPN_SUBSET=1.0

PATIENT_IDS=(s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28)

# ---- Resolve the pretrain checkpoint to fine-tune from ----
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"
if [ -z "$PRETRAIN_CKPT" ]; then
    # Folder names embed an ISO-like stamp, so a lexicographic sort + tail -1
    # selects the most recent pretrain run.
    latest="$(ls -d model_checkpoints/any2any_pretrain_3class_*/ 2>/dev/null | sort | tail -1)"
    if [ -z "$latest" ]; then
        echo "ERROR: no any2any pretrain checkpoint found under model_checkpoints/." >&2
        echo "       Run step 1 first:  bash scripts/roam_emg_1_pretrain.sh" >&2
        echo "       Or set PRETRAIN_CKPT=path/to/epoch_N.pth explicitly." >&2
        return 1 2>/dev/null || exit 1
    fi
    PRETRAIN_CKPT="${latest}epoch_11.pth"
fi
if [ ! -f "$PRETRAIN_CKPT" ]; then
    echo "ERROR: pretrain checkpoint not found: $PRETRAIN_CKPT" >&2
    echo "       Set PRETRAIN_CKPT to an existing epoch_N.pth." >&2
    return 1 2>/dev/null || exit 1
fi
echo "Fine-tuning from pretrain checkpoint: $PRETRAIN_CKPT"

for pid in "${PATIENT_IDS[@]}"; do
    echo ""
    echo "=========================================================="
    echo " ROAM-EMG step 2/3: any2any LOSO fine-tune, held-out subject = ${pid}"
    echo "=========================================================="
    python3 main.py \
        --model_choice any2any \
        --num_classes $NUM_CLASSES \
        --dataset_selection $DATASET_SELECTION \
        --window_size $WINDOW_SIZE \
        --offset $OFFSET \
        --epn_subset_percentage $EPN_SUBSET \
        --val_patient_ids "$pid" \
        --epochs $EPOCHS \
        --embedding_method linear_projection \
        --use_input_layernorm \
        --share_pe \
        --task_selection 0 1 2 \
        --saved_checkpoint_pth "$PRETRAIN_CKPT" \
        --exp_name any2any_LOSO_${pid}
done

echo ""
echo "All 28 LOSO fine-tunes complete."
echo "  Checkpoints: model_checkpoints/any2any_LOSO_<sN>_<stamp>_<host>/epoch_4.pth"
echo "  Next:        bash scripts/roam_emg_3_eval.sh"
