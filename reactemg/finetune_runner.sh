#!/bin/bash

# List of different validation patient IDs
patient_ids=("s1" "s2" "s3" "s4" "s5" "s6" "s7" "s8" "s9" "s10" "s11" "s12" "s13" "s14" "s15" "s16" "s17" "s18" "s19" "s20" "s21" "s22" "s23" "s24" "s25" "s26" "s27" "s28")

# Specify pretrain checkpoint
saved_checkpoint_pth="TODO"

# Loop over each patient ID and execute the command
for patient_id in "${patient_ids[@]}"; do
    # Constructing the exp_name dynamically based on the patient_id
    exp_name="LOSO_${patient_id}"
    
    # Construct the full command
    command="python3 main.py --offset 30 --num_classes 3 --task_selection 0 1 2 --use_input_layernorm --share_pe --dataset_selection roam_only --window_size 600 --val_patient_ids ${patient_id} --exp_name ${exp_name} --epn_subset_percentage 1.0 --epochs 5 --model_choice any2any --inner_window_size 600 --saved_checkpoint_pth ${saved_checkpoint_pth}"

    echo "Running command: $command"
    
    $command
done


