#!/usr/bin/env python3

import argparse
import csv
import os
import sys

from event_classification import main as event_classification_main

def multi_run():
    parser = argparse.ArgumentParser(
        description="Loop over a range of epochs, calling event_classification.py's main() for each epoch."
    )

    # -----------------------------------------------------------------
    # 1) Arguments for the loop itself (the checkpoint directory + epoch range)
    # -----------------------------------------------------------------
    parser.add_argument("--checkpoint_dir", required=True, type=str,
                        help="Path to a directory containing epoch_<N>.pth files.")
    parser.add_argument("--start_epoch", required=True, type=int,
                        help="Start epoch number, e.g. 0.")
    parser.add_argument("--end_epoch", required=True, type=int,
                        help="End epoch number, e.g. 19.")

    # -----------------------------------------------------------------
    # 2) Arguments passed directly into event_classification.main()
    # -----------------------------------------------------------------
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Batch size for evaluation. (Default=64)")
    parser.add_argument("--eval_task", default="predict_action", type=str,
                        help="Task to evaluate, e.g. 'predict_action'. (Default='predict_action')")
    parser.add_argument("--files_or_dirs", nargs="+", required=True,
                        help="One or more CSV files or directories (or .npy if epn_eval=1).")
    parser.add_argument("--allow_relax", default=0, type=int, choices=[0,1],
                        help="Whether to allow 'relax' logic in the event classification (Default=0)")
    parser.add_argument("--saved_checkpoint_pth", default=None, type=str,
                        help="Path to a single pretrained model. (We override this in the loop with each epoch_pth)")
    parser.add_argument("--buffer_range", default=200, type=int,
                        help="Buffer range for event classification. (Default=200)")
    parser.add_argument("--stride", default=1, type=int,
                        help="Model stride for predictions. (Default=1)")
    parser.add_argument("--lookahead", default=0, type=int,
                        help="Lookahead frames for 'future' mode. (Default=1)")
    parser.add_argument("--weight_max_factor", default=1.0, type=float,
                        help="Maximum weighting factor for the aggregator. (Default=1.0)")
    parser.add_argument("--likelihood_format", default="logits", type=str,
                        choices=["logits","probs","argmax"],
                        help="How to aggregate predictions (logits/probs/argmax). (Default='logits')")
    parser.add_argument("--samples_between_prediction", default=1, type=int,
                        help="Timesteps between aggregated predictions. (Default=1)")
    parser.add_argument("--maj_vote_range", default="single", type=str,
                        choices=["single","future"],
                        help="Aggregator on just the single current timestep or a future range. (Default='single')")
    parser.add_argument("--transition_samples_only", action='store_true',
                        help="If set, evaluate only on transition samples.")
    parser.add_argument("--mask_percentage", default=0.6, type=float,
                        help="Mask percentage for masked modeling. (Default=0.6)")
    parser.add_argument("--mask_type", default="poisson", type=str,
                        help="Type of masking for masked modeling. (Default='poisson')")
    parser.add_argument("--epn_eval", default=0, type=int, choices=[0,1],
                        help="If 1, use EPN .npy files; default=0.")
    parser.add_argument("--recog_threshold", default=0.5, type=float,
                        help="EPN recognition threshold (Default=0.5)")
    parser.add_argument("--verbose", default=0, type=int, choices=[0,1],
                        help="If 1, generate plots & per-file JSON details; if 0, skip them.")
    parser.add_argument("--model_choice", default="logits", type=str,
                        help="which model to use")
    args = parser.parse_args()

    # ------------------------------------------------------
    # CREATE OUTPUT CSV WITH CUSTOM NAMING AND LOCATION
    # ------------------------------------------------------
    #  e.g. output/looper_csv/<checkpoint_folder_name>_from_e#_to_e#.csv
    #
    checkpoint_dir = args.checkpoint_dir.rstrip("/\\")
    checkpoint_folder_name = os.path.basename(checkpoint_dir)

    start_e = args.start_epoch
    end_e   = args.end_epoch

    loop_csv_dir = "output/looper_csv"
    os.makedirs(loop_csv_dir, exist_ok=True)

    csv_output_filename = f"{checkpoint_folder_name}_from_e{start_e}_to_e{end_e}.csv"
    csv_output_path     = os.path.join(loop_csv_dir, csv_output_filename)

    # We include columns for the EPN metrics, but they will be -1 if epn_eval=0
    fieldnames = [
        "epoch",
        "event_accuracy",
        "avg_raw_accuracy",
        "epn_class_accuracy",
        "epn_smoothed_class_accuracy"
    ]

    # -----------------------------------
    # OPEN THE CSV AND WRITE THE HEADER
    # -----------------------------------
    with open(csv_output_path, mode="w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # -------------------------------
        # LOOP OVER THE EPOCH RANGE
        # -------------------------------
        for epoch in range(args.start_epoch, args.end_epoch + 1):
            epoch_filename = f"epoch_{epoch}.pth"
            checkpoint_pth = os.path.join(checkpoint_dir, epoch_filename)

            if not os.path.isfile(checkpoint_pth):
                print(f"Skipping {checkpoint_pth} (file not found).")
                continue

            print(f"\n--- Evaluating checkpoint: {checkpoint_pth} ---")

            # -----------------------------------------------------------
            # Call event_classification_main() for each epoch
            # -----------------------------------------------------------
            try:
                if args.epn_eval == 1:
                    (epoch_event_acc,
                     epoch_raw_acc,
                     epoch_epn_class_acc,
                     epoch_epn_smooth_class_acc) = event_classification_main(
                        saved_checkpoint_pth=checkpoint_pth,
                        eval_batch_size=args.eval_batch_size,
                        eval_task=args.eval_task,
                        transition_samples_only=args.transition_samples_only,
                        buffer_range=args.buffer_range,
                        mask_percentage=args.mask_percentage,
                        mask_type=args.mask_type,
                        stride=args.stride,
                        files_or_dirs=args.files_or_dirs,
                        allow_relax=args.allow_relax,
                        lookahead=args.lookahead,
                        weight_max_factor=args.weight_max_factor,
                        likelihood_format=args.likelihood_format,
                        samples_between_prediction=args.samples_between_prediction,
                        maj_vote_range=args.maj_vote_range,
                        epn_eval=args.epn_eval,
                        recog_threshold=args.recog_threshold,
                        verbose=args.verbose,
                        model_choice=args.model_choice
                    )
                else:
                    (epoch_event_acc,
                     epoch_raw_acc) = event_classification_main(
                        saved_checkpoint_pth=checkpoint_pth,
                        eval_batch_size=args.eval_batch_size,
                        eval_task=args.eval_task,
                        transition_samples_only=args.transition_samples_only,
                        buffer_range=args.buffer_range,
                        mask_percentage=args.mask_percentage,
                        mask_type=args.mask_type,
                        stride=args.stride,
                        files_or_dirs=args.files_or_dirs,
                        allow_relax=args.allow_relax,
                        lookahead=args.lookahead,
                        weight_max_factor=args.weight_max_factor,
                        likelihood_format=args.likelihood_format,
                        samples_between_prediction=args.samples_between_prediction,
                        maj_vote_range=args.maj_vote_range,
                        epn_eval=args.epn_eval,
                        recog_threshold=args.recog_threshold,
                        verbose=args.verbose,
                        model_choice=args.model_choice
                    )
                    # Dummy placeholders for non-EPN mode
                    epoch_epn_class_acc = -1.0
                    epoch_epn_smooth_class_acc = -1.0

            except Exception as e:
                raise Exception(f"Error while calling event_classification_main on {checkpoint_pth}: {str(e)}")

            # -------------------------------
            # WRITE RESULTS TO THE CSV
            # -------------------------------
            writer.writerow({
                "epoch": epoch,
                "event_accuracy": epoch_event_acc,
                "avg_raw_accuracy": epoch_raw_acc,
                "epn_class_accuracy": epoch_epn_class_acc,
                "epn_smoothed_class_accuracy": epoch_epn_smooth_class_acc
            })

            # Print them to console too
            print(f"Epoch {epoch}: event_accuracy = {epoch_event_acc:.4f}, raw_accuracy = {epoch_raw_acc:.4f}")
            if args.epn_eval == 1:
                print(f"           epn_class_acc = {epoch_epn_class_acc:.4f}, epn_smooth_acc = {epoch_epn_smooth_class_acc:.4f}")

    print(f"\nAll done. Results saved to: {csv_output_path}")


if __name__ == "__main__":
    multi_run()
