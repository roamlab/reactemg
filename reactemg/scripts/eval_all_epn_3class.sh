#!/bin/bash
# Evaluate every EPN-trained 3-class model produced by train_all_epn_3class.sh.
#
# Fill in the six checkpoint paths below (relative to reactemg/, e.g.
# "model_checkpoints/ann_epn_3class_<stamp>_<host>/epoch_12.pth"), then run:
#   source scripts/eval_all_epn_3class.sh   # from reactemg/
#   bash   scripts/eval_all_epn_3class.sh   # from anywhere
# The script `cd`s to reactemg/ so checkpoint paths and output/ work the same
# regardless of where it's invoked from.
#
# any2any, lstm, ed_tcn  -> evaluated TWICE: no-lookahead and with-lookahead.
# ann, trahgr, lda       -> evaluated ONCE: no-lookahead only.
#
# Leave any path as "" (empty string) to skip that model.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

# ============================================================
#                      FILL THESE IN
# ============================================================
CKPT_ANY2ANY="TODO"
CKPT_ANN="model_checkpoints/ann_epn_3class_2026-05-26_09-35-05_pc1/epoch_12.pth"
CKPT_LSTM="model_checkpoints/lstm_epn_3class_2026-05-26_10-34-31_pc1/epoch_12.pth"
CKPT_ED_TCN="model_checkpoints/ed_tcn_epn_3class_2026-05-26_11-01-41_pc1/epoch_12.pth"
CKPT_TRAHGR="TODO"
CKPT_LDA="TODO"
# ============================================================

eval_no_la() {
    local model=$1
    local ckpt=$2
    echo ""
    echo "=========================================================="
    echo " Evaluating ${model} (NO lookahead)"
    echo " Checkpoint: ${ckpt}"
    echo "=========================================================="
    python3 event_classification.py \
        --saved_checkpoint_pth "$ckpt" \
        --model_choice "$model" \
        --eval_task predict_action \
        --files_or_dirs ../data/EMG-EPN-612 \
        --epn_eval 1 \
        --buffer_range 200 \
        --allow_relax 0 \
        --stride 20 \
        --lookahead 0 \
        --samples_between_prediction 1 \
        --maj_vote_range single \
        --weight_max_factor 1.0 \
        --likelihood_format logits \
        --verbose 0
}

eval_with_la() {
    local model=$1
    local ckpt=$2
    echo ""
    echo "=========================================================="
    echo " Evaluating ${model} (WITH lookahead 50)"
    echo " Checkpoint: ${ckpt}"
    echo "=========================================================="
    python3 event_classification.py \
        --saved_checkpoint_pth "$ckpt" \
        --model_choice "$model" \
        --eval_task predict_action \
        --files_or_dirs ../data/EMG-EPN-612 \
        --epn_eval 1 \
        --buffer_range 200 \
        --allow_relax 0 \
        --stride 1 \
        --lookahead 50 \
        --samples_between_prediction 20 \
        --maj_vote_range future \
        --weight_max_factor 1.0 \
        --likelihood_format logits \
        --verbose 0
}

# Helper: skip if path is empty or still set to "TODO"
should_run() {
    local p=$1
    [ -n "$p" ] && [ "$p" != "TODO" ]
}

# ---------- any2any: both ----------
if should_run "$CKPT_ANY2ANY"; then
    eval_no_la   any2any "$CKPT_ANY2ANY"
    eval_with_la any2any "$CKPT_ANY2ANY"
else
    echo "[skip] any2any (CKPT_ANY2ANY not set)"
fi

# ---------- ann: no-lookahead only ----------
if should_run "$CKPT_ANN"; then
    eval_no_la ann "$CKPT_ANN"
else
    echo "[skip] ann (CKPT_ANN not set)"
fi

# ---------- lstm: both ----------
if should_run "$CKPT_LSTM"; then
    eval_no_la   lstm "$CKPT_LSTM"
    eval_with_la lstm "$CKPT_LSTM"
else
    echo "[skip] lstm (CKPT_LSTM not set)"
fi

# ---------- ed_tcn: both ----------
if should_run "$CKPT_ED_TCN"; then
    eval_no_la   ed_tcn "$CKPT_ED_TCN"
    eval_with_la ed_tcn "$CKPT_ED_TCN"
else
    echo "[skip] ed_tcn (CKPT_ED_TCN not set)"
fi

# ---------- trahgr: no-lookahead only ----------
if should_run "$CKPT_TRAHGR"; then
    eval_no_la trahgr "$CKPT_TRAHGR"
else
    echo "[skip] trahgr (CKPT_TRAHGR not set)"
fi

# ---------- lda: no-lookahead only ----------
if should_run "$CKPT_LDA"; then
    eval_no_la lda "$CKPT_LDA"
else
    echo "[skip] lda (CKPT_LDA not set)"
fi

echo ""
echo "All 3-class EPN evals complete."
