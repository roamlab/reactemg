#!/bin/bash
# LOSO fine-tune driver for ROAM-EMG across all neural model variants
# (any2any, ann, lstm, ed_tcn, trahgr). For each architecture whose pretrain
# checkpoint is supplied below, this script trains a separate model for every
# subject s1..s28 in ROAM-EMG, holding that subject out for validation.
#
# Fill in the five checkpoint paths (each relative to reactemg/, e.g.
# "model_checkpoints/any2any_pretrain_3class_<stamp>_<host>/epoch_11.pth"),
# then run:
#   source scripts/finetune_all_loso.sh   # from reactemg/
#   bash   scripts/finetune_all_loso.sh   # from anywhere
# The script `cd`s to reactemg/ so relative checkpoint paths and the
# model_checkpoints/, output/, wandb/ folders all resolve consistently.
#
# Leave a checkpoint as "" or "TODO" to skip that architecture.
#
# NOTE: LDA has no warm-start (sklearn LinearDiscriminantAnalysis), so its LOSO
# "fine-tune" is just a fresh per-subject fit on the ROAM training fold — no
# pretrain checkpoint is used. It is included below, gated by RUN_LDA (default 0)
# instead of a checkpoint path. Enabling it does not affect the other models.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

# ============================================================
#                  PRETRAIN CHECKPOINTS — FILL IN
# ============================================================
CKPT_ANY2ANY="TODO"
CKPT_ANN="TODO"
CKPT_LSTM="TODO"
CKPT_ED_TCN="TODO"
CKPT_TRAHGR="TODO"

# ============================================================

# LDA uses no pretrain checkpoint (no warm-start); its LOSO fit is standalone.
# Set to 1 to also run per-subject LDA fits on ROAM; 0 skips it.
RUN_LDA="${RUN_LDA:-0}"

NUM_CLASSES=3
DATASET_SELECTION=roam_only
WINDOW_SIZE=600
# ED-TCN's autoencoder needs T_red divisible by 4 (= (W-150)/25 + 1). W=625
# yields T_red=20; matches the pretrain script.
ED_TCN_WINDOW=625
OFFSET=30
EPOCHS=5
# epn_subset_percentage is required by the parser but unused for roam_only.
EPN_SUBSET=1.0

PATIENT_IDS=("s1" "s2" "s3" "s4" "s5" "s6" "s7" "s8" "s9" "s10" "s11" "s12" "s13" "s14" "s15" "s16" "s17" "s18" "s19" "s20" "s21" "s22" "s23" "s24" "s25" "s26" "s27" "s28")

# Helper: skip if path is empty or still set to "TODO"
should_run() {
    local p=$1
    [ -n "$p" ] && [ "$p" != "TODO" ]
}

# ---------- per-architecture fine-tune functions ----------
# Each runs one (architecture, held-out subject) pair.
ft_any2any() {
    local pid=$1
    local ckpt=$2
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
        --saved_checkpoint_pth "$ckpt" \
        --exp_name any2any_LOSO_${pid}
}

ft_ann() {
    local pid=$1
    local ckpt=$2
    python3 main.py \
        --model_choice ann \
        --num_classes $NUM_CLASSES \
        --dataset_selection $DATASET_SELECTION \
        --window_size $WINDOW_SIZE \
        --offset $OFFSET \
        --epn_subset_percentage $EPN_SUBSET \
        --val_patient_ids "$pid" \
        --epochs $EPOCHS \
        --saved_checkpoint_pth "$ckpt" \
        --exp_name ann_LOSO_${pid}
}

ft_lstm() {
    local pid=$1
    local ckpt=$2
    python3 main.py \
        --model_choice lstm \
        --num_classes $NUM_CLASSES \
        --dataset_selection $DATASET_SELECTION \
        --window_size $WINDOW_SIZE \
        --offset $OFFSET \
        --epn_subset_percentage $EPN_SUBSET \
        --val_patient_ids "$pid" \
        --epochs $EPOCHS \
        --saved_checkpoint_pth "$ckpt" \
        --exp_name lstm_LOSO_${pid}
}

ft_ed_tcn() {
    local pid=$1
    local ckpt=$2
    # ED-TCN uses window_size=625 (T_red=20, divisible by 4); same as pretrain.
    python3 main.py \
        --model_choice ed_tcn \
        --num_classes $NUM_CLASSES \
        --dataset_selection $DATASET_SELECTION \
        --window_size $ED_TCN_WINDOW \
        --offset $OFFSET \
        --epn_subset_percentage $EPN_SUBSET \
        --val_patient_ids "$pid" \
        --epochs $EPOCHS \
        --saved_checkpoint_pth "$ckpt" \
        --exp_name ed_tcn_LOSO_${pid}
}

ft_trahgr() {
    local pid=$1
    local ckpt=$2
    # TraHGR-Huge (D=144, h=8, L=1) — must match the pretrain config so the
    # saved args_dict mirrors what's actually constructed; event_classification.py
    # rebuilds TraHGR from args_dict at eval time.
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
        --saved_checkpoint_pth "$ckpt" \
        --exp_name trahgr_LOSO_${pid}
}

ft_lda() {
    local pid=$1
    # LDA has no warm-start: this is a fresh fit on the ROAM LOSO training fold
    # (all subjects except "$pid"), NOT a fine-tune. No --saved_checkpoint_pth,
    # and --epochs 1 (the fit is deterministic; eval expects epoch_1.pth).
    python3 main.py \
        --model_choice lda \
        --num_classes $NUM_CLASSES \
        --dataset_selection $DATASET_SELECTION \
        --window_size $WINDOW_SIZE \
        --offset $OFFSET \
        --epn_subset_percentage $EPN_SUBSET \
        --val_patient_ids "$pid" \
        --epochs 1 \
        --exp_name lda_LOSO_${pid}
}

# ---------- LOSO loops, one architecture at a time ----------

if should_run "$CKPT_ANY2ANY"; then
    echo ""
    echo "############################################################"
    echo " any2any LOSO fine-tune (checkpoint: $CKPT_ANY2ANY)"
    echo "############################################################"
    for pid in "${PATIENT_IDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo " any2any LOSO subject = ${pid}"
        echo "=========================================================="
        ft_any2any "$pid" "$CKPT_ANY2ANY"
    done
else
    echo "[skip] any2any LOSO (CKPT_ANY2ANY not set)"
fi

if should_run "$CKPT_ANN"; then
    echo ""
    echo "############################################################"
    echo " ann LOSO fine-tune (checkpoint: $CKPT_ANN)"
    echo "############################################################"
    for pid in "${PATIENT_IDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo " ann LOSO subject = ${pid}"
        echo "=========================================================="
        ft_ann "$pid" "$CKPT_ANN"
    done
else
    echo "[skip] ann LOSO (CKPT_ANN not set)"
fi

if should_run "$CKPT_LSTM"; then
    echo ""
    echo "############################################################"
    echo " lstm LOSO fine-tune (checkpoint: $CKPT_LSTM)"
    echo "############################################################"
    for pid in "${PATIENT_IDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo " lstm LOSO subject = ${pid}"
        echo "=========================================================="
        ft_lstm "$pid" "$CKPT_LSTM"
    done
else
    echo "[skip] lstm LOSO (CKPT_LSTM not set)"
fi

if should_run "$CKPT_ED_TCN"; then
    echo ""
    echo "############################################################"
    echo " ed_tcn LOSO fine-tune (checkpoint: $CKPT_ED_TCN)"
    echo "############################################################"
    for pid in "${PATIENT_IDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo " ed_tcn LOSO subject = ${pid}"
        echo "=========================================================="
        ft_ed_tcn "$pid" "$CKPT_ED_TCN"
    done
else
    echo "[skip] ed_tcn LOSO (CKPT_ED_TCN not set)"
fi

if should_run "$CKPT_TRAHGR"; then
    echo ""
    echo "############################################################"
    echo " trahgr LOSO fine-tune (checkpoint: $CKPT_TRAHGR)"
    echo "############################################################"
    for pid in "${PATIENT_IDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo " trahgr LOSO subject = ${pid}"
        echo "=========================================================="
        ft_trahgr "$pid" "$CKPT_TRAHGR"
    done
else
    echo "[skip] trahgr LOSO (CKPT_TRAHGR not set)"
fi

if [ "$RUN_LDA" = "1" ]; then
    echo ""
    echo "############################################################"
    echo " lda LOSO fit (fresh per-subject fit; no pretrain checkpoint)"
    echo "############################################################"
    for pid in "${PATIENT_IDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo " lda LOSO subject = ${pid}"
        echo "=========================================================="
        ft_lda "$pid"
    done
else
    echo "[skip] lda LOSO (set RUN_LDA=1 to enable)"
fi

echo ""
echo "All LOSO fine-tunes complete."
