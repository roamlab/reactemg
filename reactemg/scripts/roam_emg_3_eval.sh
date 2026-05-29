#!/bin/bash
# ROAM-EMG pipeline — STEP 3 of 3: evaluate.
#
# Evaluates the 28 any2any LOSO checkpoints from step 2
# (scripts/roam_emg_2_finetune.sh) and reports the per-subject mean +/- std of
# transition accuracy and raw accuracy across subjects s1..s28.
#
# any2any is run TWICE via eval_runner.py:
#   * no-lookahead   — online prediction with no future smoothing
#   * with-lookahead — 50-sample future majority vote (the paper's online config)
#
# For each subject it auto-discovers the MOST RECENT
# any2any_LOSO_<sN>_<stamp>_<host> folder and requires epoch_4.pth (any2any
# fine-tunes for --epochs 5 and is 0-indexed, so the last epoch is epoch_4).
#
# This is the SINGLE-MODEL pipeline for reproducing the main method. To evaluate
# every baseline as well, use the broader scripts/eval_all_loso.sh instead.
#
# Invoke from anywhere; the script `cd`s to reactemg/:
#   bash scripts/roam_emg_3_eval.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}/.."

EPOCH_FILE=epoch_4.pth
PATIENT_IDS=(s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 s12 s13 s14 s15 s16 s17 s18 s19 s20 s21 s22 s23 s24 s25 s26 s27 s28)

# Auto-discover one finetune folder per subject (latest run, requires epoch file).
folders=()
missing=()
for pid in "${PATIENT_IDS[@]}"; do
    # Trailing "_" after the subject id stops s1 from matching s10/s12/etc.
    latest="$(ls -d model_checkpoints/any2any_LOSO_${pid}_*/ 2>/dev/null | sort | tail -1)"
    if [ -z "$latest" ]; then
        missing+=("$pid")
        continue
    fi
    folder="$(basename "$latest")"
    if [ ! -f "model_checkpoints/${folder}/${EPOCH_FILE}" ]; then
        missing+=("${pid}(no ${EPOCH_FILE})")
        continue
    fi
    folders+=("$folder")
done

if [ "${#folders[@]}" -ne "${#PATIENT_IDS[@]}" ]; then
    echo "ERROR: found ${#folders[@]}/${#PATIENT_IDS[@]} any2any LOSO checkpoints (epoch=${EPOCH_FILE})." >&2
    echo "       missing: ${missing[*]:-none}" >&2
    echo "       Run step 2 first:  bash scripts/roam_emg_2_finetune.sh" >&2
    return 1 2>/dev/null || exit 1
fi

echo ""
echo "############################################################"
echo " ROAM-EMG step 3/3: any2any eval (no lookahead) — ${#folders[@]} subjects"
echo "############################################################"
python3 eval_runner.py \
    --model_choice any2any \
    --epoch_file "$EPOCH_FILE" \
    --output_tag "any2any_" \
    --lookahead_mode no_lookahead \
    --folders "${folders[@]}"

echo ""
echo "############################################################"
echo " ROAM-EMG step 3/3: any2any eval (with lookahead 50) — ${#folders[@]} subjects"
echo "############################################################"
python3 eval_runner.py \
    --model_choice any2any \
    --epoch_file "$EPOCH_FILE" \
    --output_tag "any2any_" \
    --lookahead_mode with_lookahead \
    --folders "${folders[@]}"

echo ""
echo "=========================================================="
echo " ROAM-EMG eval complete."
echo "   Aggregated summaries:"
echo "     output/batch_eval_summary_any2any_LA0_<stamp>.{txt,json}   (no lookahead)"
echo "     output/batch_eval_summary_any2any_LA50_<stamp>.{txt,json}  (with lookahead)"
echo "   Per-subject details & plots:"
echo "     output/any2any_LOSO_<sN>_<stamp>_<host>_epoch_4_LA{0,50}/"
echo "=========================================================="
