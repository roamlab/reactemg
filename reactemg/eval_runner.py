#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import event_classification as ec

# ------------------------------------------------------------------
# Copy-paste your list of run-folders
# Shortcut for generating a copy-paste-friendly folder list:
# find . -maxdepth 1 -type d -name '*LOSO*' -printf '"%P",\n'
# ------------------------------------------------------------------
model_folders = [
    # TODO
]

# No lookahead
COMMON_KWARGS = dict(
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
    recog_threshold=0.5,
    verbose=1,
    model_choice="any2any",
    sample_range=None,
)


"""
# With lookahead
COMMON_KWARGS = dict(
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
    recog_threshold=0.5,
    verbose=1,
    model_choice="any2any",
    sample_range=None,
)
"""

event_accs = []
raw_accs = []

for folder in model_folders:
    ckpt = Path("model_checkpoints") / folder / "epoch_4.pth"
    print(f"→ evaluating {ckpt}")

    evt_acc, raw_acc = ec.main(
        saved_checkpoint_pth=str(ckpt),
        **COMMON_KWARGS,
    )

    event_accs.append(evt_acc)
    raw_accs.append(raw_acc)

event_mean = np.mean(event_accs)
event_std = np.std(event_accs, ddof=1)
raw_mean = np.mean(raw_accs)
raw_std = np.std(raw_accs, ddof=1)

print("\n=========== FINAL SUMMARY ===========")
print(f"Subjects evaluated : {len(model_folders)}")
print(f"Transition Accuracy (μ±σ): {event_mean:.4f} ± {event_std:.4f}")
print(f"Raw Accuracy        (μ±σ): {raw_mean:.4f} ± {raw_std:.4f}")
print("=====================================")
