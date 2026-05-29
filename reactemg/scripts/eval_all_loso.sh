#!/bin/bash
# Batch-evaluate the ROAM-EMG LOSO finetuned checkpoints for every model variant,
# producing per-model aggregated summaries (mean +/- per-subject sample std of
# transition accuracy and raw accuracy over the 28 LOSO subjects).
#
# Evaluation matrix:
#   no-lookahead : run for ALL variants (lda, ann, lstm, ed_tcn, trahgr, any2any)
#   with-lookahead: run ONLY for the dense per-timestep models (any2any, lstm, ed_tcn),
#                   since lookahead aggregation (future majority vote over windows)
#                   only changes those models' outputs.
# So any2any/lstm/ed_tcn get TWO summaries each (LA0 + LA50); ann/trahgr/lda get one (LA0).
#
# For each model variant it:
#   1. Auto-discovers, per subject s1..s28, the MOST RECENT finetune folder
#      named "<model>_LOSO_<sN>_<stamp>_<host>" under model_checkpoints/.
#   2. Requires the chosen epoch file to exist in each folder.
#   3. If all 28 subjects are found  -> runs eval_runner.py for that model.
#      If any are missing            -> SKIPS the model and reports what's missing.
#
# Epoch per model (last saved epoch given the finetune --epochs):
#   any2any -> epoch_4.pth   (0-indexed: epoch_0..epoch_4)
#   ann/lstm/ed_tcn/trahgr -> epoch_5.pth   (1-indexed: epoch_1..epoch_5)
#   lda -> epoch_1.pth       (sklearn fit; --epochs 1)
#
# Invoke from anywhere; the script `cd`s to reactemg/:
#   source scripts/eval_all_loso.sh   # from reactemg/
#   bash   scripts/eval_all_loso.sh   # from anywhere
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

# All model variants to consider. Variants without (complete) checkpoints are
# auto-skipped and reported below, so it's fine to list ones you haven't trained.
MODELS=(any2any ann lstm ed_tcn trahgr lda)

# Dense per-timestep models that additionally get a with-lookahead run.
LOOKAHEAD_MODELS=(any2any lstm ed_tcn)

PATIENT_IDS=(s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28)

# Map model -> last-epoch checkpoint filename.
epoch_for_model() {
    case "$1" in
        any2any) echo "epoch_4.pth" ;;
        lda)     echo "epoch_1.pth" ;;
        *)       echo "epoch_5.pth" ;;   # ann, lstm, ed_tcn, trahgr
    esac
}

# Does this model also get a with-lookahead run?
in_lookahead_set() {
    local m=$1
    for x in "${LOOKAHEAD_MODELS[@]}"; do
        [ "$x" = "$m" ] && return 0
    done
    return 1
}

ran_models=()
skipped_models=()

for model in "${MODELS[@]}"; do
    epoch_file="$(epoch_for_model "$model")"
    folders=()
    missing=()

    for pid in "${PATIENT_IDS[@]}"; do
        # Most-recent folder for this (model, subject): the trailing "_" after
        # the subject id prevents s1 from matching s10/s12/etc. Folder names embed
        # an ISO-like timestamp, so a lexicographic sort + tail -1 = latest run.
        latest="$(ls -d model_checkpoints/${model}_LOSO_${pid}_*/ 2>/dev/null | sort | tail -1)"
        if [ -z "$latest" ]; then
            missing+=("$pid")
            continue
        fi
        folder="$(basename "$latest")"
        if [ ! -f "model_checkpoints/${folder}/${epoch_file}" ]; then
            missing+=("${pid}(no ${epoch_file})")
            continue
        fi
        folders+=("$folder")
    done

    if [ "${#folders[@]}" -ne "${#PATIENT_IDS[@]}" ]; then
        echo ""
        echo "[SKIP] ${model}: found ${#folders[@]}/${#PATIENT_IDS[@]} checkpoints (epoch=${epoch_file})."
        echo "       missing: ${missing[*]:-none}"
        skipped_models+=("$model")
        continue
    fi

    # ---- no-lookahead run (all variants) ----
    echo ""
    echo "############################################################"
    echo " [RUN] ${model} (no lookahead): ${#folders[@]}/${#PATIENT_IDS[@]} ckpts (epoch=${epoch_file})"
    echo "############################################################"
    python3 eval_runner.py \
        --model_choice "$model" \
        --epoch_file "$epoch_file" \
        --output_tag "${model}_" \
        --lookahead_mode no_lookahead \
        --folders "${folders[@]}"

    # ---- with-lookahead run (dense models only) ----
    if in_lookahead_set "$model"; then
        echo ""
        echo "############################################################"
        echo " [RUN] ${model} (with lookahead): ${#folders[@]}/${#PATIENT_IDS[@]} ckpts (epoch=${epoch_file})"
        echo "############################################################"
        python3 eval_runner.py \
            --model_choice "$model" \
            --epoch_file "$epoch_file" \
            --output_tag "${model}_" \
            --lookahead_mode with_lookahead \
            --folders "${folders[@]}"
    fi

    ran_models+=("$model")
done

echo ""
echo "=========================================================="
echo " eval_all_loso complete"
echo "   evaluated : ${ran_models[*]:-none}"
echo "   skipped   : ${skipped_models[*]:-none}"
echo " Summaries: output/batch_eval_summary_<model>_LA0_<stamp>.{txt,json}   (no lookahead, all models)"
echo "            output/batch_eval_summary_<model>_LA50_<stamp>.{txt,json}  (lookahead: any2any/lstm/ed_tcn)"
echo "=========================================================="
