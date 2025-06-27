#!/usr/bin/env bash

# -----------------------------------------------------------------------------
# List of model folder names
# -----------------------------------------------------------------------------
model_folders=(
"FOLDER_1"
"FOLDER_2"
"FOLDER_3"
# ...
)

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
for folder in "${model_folders[@]}"; do
  echo "Running command for model folder: $folder"

  python event_loop.py \
      --eval_task predict_action \
      --buffer_range 200 \
      --files_or_dirs ../data/s1 \
                      ../data/s2 \
                      ../data/s3 \
                      ../data/s4 \
                      ../data/s5 \
                      ../data/s6 \
                      ../data/s7 \
                      ../data/s8 \
                      ../data/s9 \
                      ../data/s10 \
                      ../data/s11 \
                      ../data/s12 \
                      ../data/s13 \
                      ../data/s14 \
                      ../data/s15 \
                      ../data/s16 \
                      ../data/s17 \
                      ../data/s18 \
                      ../data/s19 \
                      ../data/s20 \
                      ../data/s21 \
                      ../data/s22 \
                      ../data/s23 \
                      ../data/s24 \
                      ../data/s25 \
                      ../data/s26 \
                      ../data/s27 \
                      ../data/s28 \
      --allow_relax 0 \
      --stride 20 \
      --lookahead 0 \
      --weight_max_factor 1.0 \
      --likelihood_format logits \
      --samples_between_prediction 1 \
      --maj_vote_range future \
      --checkpoint_dir "model_checkpoints/$folder" \
      --start_epoch 4 \
      --end_epoch 4 \
      --model_choice any2any

  echo "Finished command for model folder: $folder"
  echo "-----------------------------------------"
done
