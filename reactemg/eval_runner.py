#!/usr/bin/env python3
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse
import json
import numpy as np
import re
import event_classification as ec

# ------------------------------------------------------------------
# Copy-paste your list of run-folders
# Shortcut for generating a copy-paste-friendly folder list:
# find . -maxdepth 1 -type d -name '*LOSO*' -printf '"%P",\n'
# ------------------------------------------------------------------
model_folders = [
]

# No lookahead
COMMON_KWARGS_NO_LOOKAHEAD = dict(
    eval_batch_size=64,
    eval_task="predict_action",
    transition_samples_only=False,
    buffer_range=200,
    mask_percentage=0.6,
    mask_type="poisson",
    stride=20,
    files_or_dirs=["../data/ROAM_EMG"],
    allow_relax=0,
    lookahead=0,
    weight_max_factor=1.0,
    likelihood_format="logits",
    samples_between_prediction=1,
    maj_vote_range="single",
    epn_eval=0,
    verbose=1,
    model_choice="any2any",
    sample_range=None,
    strict_transition=0,
)

# With lookahead
COMMON_KWARGS_WITH_LOOKAHEAD = dict(
    eval_batch_size=64,
    eval_task="predict_action",
    transition_samples_only=False,
    buffer_range=200,
    mask_percentage=0.6,
    mask_type="poisson",
    stride=1,
    files_or_dirs=["../data/ROAM_EMG"],
    allow_relax=0,
    lookahead=50,
    weight_max_factor=1.0,
    likelihood_format="logits",
    samples_between_prediction=20,
    maj_vote_range="future",
    epn_eval=0,
    verbose=1,
    model_choice="any2any",
    sample_range=None,
    strict_transition=0,
)

# Default (used when eval_runner.py is run with no --lookahead_mode): with lookahead,
# matching the previously-active config.
COMMON_KWARGS = COMMON_KWARGS_WITH_LOOKAHEAD


def extract_subject_id(folder_name):
    """Extract subject ID (e.g., 's1', 's12') from folder name."""
    match = re.search(r"LOSO_(s\d+)", folder_name)
    return match.group(1) if match else folder_name


def main(folders=None, model_choice=None, epoch_file="epoch_4.pth", output_tag="",
         common_kwargs=None):
    # Fall back to the module-level list / config when called with no args
    # (preserves the original `python3 eval_runner.py` behavior).
    if folders is None:
        folders = model_folders
    if common_kwargs is None:
        common_kwargs = COMMON_KWARGS
    kwargs = dict(common_kwargs)
    if model_choice is not None:
        kwargs["model_choice"] = model_choice
    model_choice = kwargs["model_choice"]

    # Per-subject tracking
    per_subject_results = []

    # Aggregate failure reasons across all subjects
    aggregate_reasons = Counter()

    for folder in folders:
        ckpt = Path("model_checkpoints") / folder / epoch_file
        subject_id = extract_subject_id(folder)
        print(f"→ evaluating {subject_id}: {ckpt}")

        evt_acc, raw_acc, reason_counter = ec.main(
            saved_checkpoint_pth=str(ckpt),
            **kwargs,
        )

        # Store per-subject results
        per_subject_results.append({
            "subject_id": subject_id,
            "folder": folder,
            "transition_accuracy": evt_acc,
            "raw_accuracy": raw_acc,
            "failure_reasons": dict(reason_counter),
        })

        # Aggregate failure reasons
        aggregate_reasons.update(reason_counter)

    # Compute statistics
    event_accs = [r["transition_accuracy"] for r in per_subject_results]
    raw_accs = [r["raw_accuracy"] for r in per_subject_results]

    event_mean = np.mean(event_accs)
    event_std = np.std(event_accs, ddof=1)
    raw_mean = np.mean(raw_accs)
    raw_std = np.std(raw_accs, ddof=1)

    # Print summary to console
    print("\n=========== FINAL SUMMARY ===========")
    print(f"Model              : {model_choice}")
    print(f"Subjects evaluated : {len(folders)}")
    print(f"Transition Accuracy (μ±σ): {event_mean:.4f} ± {event_std:.4f}")
    print(f"Raw Accuracy        (μ±σ): {raw_mean:.4f} ± {raw_std:.4f}")
    print("\n--- Aggregated Failure Categories ---")
    total_transitions = sum(aggregate_reasons.values())
    for reason, count in sorted(aggregate_reasons.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_transitions if total_transitions > 0 else 0
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print("=====================================")

    # Save comprehensive summary to file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    lookahead = common_kwargs.get("lookahead", 0)
    summary_dir = Path("output")
    summary_dir.mkdir(exist_ok=True)

    # Text summary
    strict_mode = common_kwargs.get("strict_transition", 0)
    txt_path = summary_dir / f"batch_eval_summary_{output_tag}LA{lookahead}_{timestamp}.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BATCH LOSO EVALUATION SUMMARY\n")
        f.write(f"Model: {model_choice}\n")
        f.write(f"Epoch file: {epoch_file}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Lookahead: {lookahead}\n")
        f.write(f"Strict Transition: {bool(strict_mode)}\n")
        f.write("=" * 60 + "\n\n")

        f.write("=== Overall Statistics (per-subject mean ± sample std) ===\n")
        f.write(f"Subjects evaluated: {len(folders)}\n")
        f.write(f"Transition Accuracy (μ±σ): {event_mean:.4f} ± {event_std:.4f}\n")
        f.write(f"Raw Accuracy        (μ±σ): {raw_mean:.4f} ± {raw_std:.4f}\n")
        f.write(f"Total Transitions: {total_transitions}\n\n")

        f.write("=== Aggregated Failure Categories ===\n")
        for reason, count in sorted(aggregate_reasons.items(), key=lambda x: -x[1]):
            pct = 100 * count / total_transitions if total_transitions > 0 else 0
            f.write(f"  {reason}: {count} ({pct:.1f}%)\n")
        f.write("\n")

        f.write("=== Per-Subject Results ===\n")
        f.write(f"{'Subject':<10} {'Trans Acc':>12} {'Raw Acc':>12}\n")
        f.write("-" * 36 + "\n")
        for r in per_subject_results:
            f.write(f"{r['subject_id']:<10} {r['transition_accuracy']:>12.4f} {r['raw_accuracy']:>12.4f}\n")
        f.write("\n")

        f.write("=== Per-Subject Failure Breakdown ===\n")
        for r in per_subject_results:
            f.write(f"\n{r['subject_id']}:\n")
            subject_total = sum(r["failure_reasons"].values())
            for reason, count in sorted(r["failure_reasons"].items(), key=lambda x: -x[1]):
                pct = 100 * count / subject_total if subject_total > 0 else 0
                f.write(f"    {reason}: {count} ({pct:.1f}%)\n")

        f.write("\nEnd of summary.\n")

    print(f"\nText summary saved to: {txt_path}")

    # JSON summary (for programmatic access)
    json_path = summary_dir / f"batch_eval_summary_{output_tag}LA{lookahead}_{timestamp}.json"
    strict_transition = common_kwargs.get("strict_transition", 0)
    json_data = {
        "timestamp": timestamp,
        "model_choice": model_choice,
        "epoch_file": epoch_file,
        "config": {
            "lookahead": lookahead,
            "buffer_range": common_kwargs.get("buffer_range"),
            "stride": common_kwargs.get("stride"),
            "allow_relax": common_kwargs.get("allow_relax"),
            "maj_vote_range": common_kwargs.get("maj_vote_range"),
            "strict_transition": strict_transition,
        },
        "overall": {
            "num_subjects": len(folders),
            "transition_accuracy_mean": event_mean,
            "transition_accuracy_std": event_std,
            "raw_accuracy_mean": raw_mean,
            "raw_accuracy_std": raw_std,
            "total_transitions": total_transitions,
        },
        "aggregated_failure_reasons": dict(aggregate_reasons),
        "per_subject_results": per_subject_results,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"JSON summary saved to: {json_path}")

    return event_mean, event_std, raw_mean, raw_std


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="LOSO checkpoint folder names under model_checkpoints/. "
             "If omitted, uses the module-level model_folders list.",
    )
    parser.add_argument(
        "--model_choice",
        default=None,
        help="Architecture to evaluate; overrides COMMON_KWARGS['model_choice'].",
    )
    parser.add_argument(
        "--epoch_file",
        default="epoch_4.pth",
        help="Checkpoint filename within each LOSO folder (e.g. epoch_4.pth).",
    )
    parser.add_argument(
        "--output_tag",
        default="",
        help="Prefix inserted into output summary filenames (e.g. 'any2any_').",
    )
    parser.add_argument(
        "--lookahead_mode",
        default="with_lookahead",
        choices=["no_lookahead", "with_lookahead"],
        help="Which smoothing config to use (default: with_lookahead).",
    )
    args = parser.parse_args()
    common_kwargs = (
        COMMON_KWARGS_NO_LOOKAHEAD
        if args.lookahead_mode == "no_lookahead"
        else COMMON_KWARGS_WITH_LOOKAHEAD
    )
    main(args.folders, args.model_choice, args.epoch_file, args.output_tag, common_kwargs)
