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
#TODO
]

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

print("\n=========== FINAL SUMMARY ===========")
print(f"Models evaluated   : {len(model_folders)}")
print(f"Average Transition Accuracy: {np.mean(event_accs):.4f}")
print(f"Average Raw Accuracy  : {np.mean(raw_accs):.4f}")
print("=====================================")
