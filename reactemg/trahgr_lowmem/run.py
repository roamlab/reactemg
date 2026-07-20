"""
Drop-in launcher for TraHGR training that uses the low-memory (lazy) TraHGR_Dataset.

Use it EXACTLY like `python3 main.py ...` — same flags, same behavior — e.g.:

    python3 trahgr_lowmem/run.py --model_choice trahgr --num_classes 3 \
        --dataset_selection pub_with_epn --window_size 600 --offset 30 \
        --epn_subset_percentage 1.0 --val_patient_ids s1 \
        --embedding_dim 144 --nhead 8 --num_layers 1 --batch_size 64 \
        --exp_name trahgr_pretrain_3class

It swaps the TraHGR dataset class in-process (before main.py runs) and then delegates to
the stock main.py. The stock source files (main.py, dataset.py, preprocessing_utils.py,
...) are NEVER modified, and no other model variant is affected. Only TraHGR runs use the
lazy loader; every returned sample is identical to the eager version (see lowmem_dataset.py).

Note: only the TRAINING path OOMs (it loads the whole pool at once). Evaluation builds the
TraHGR dataset one file at a time, so it stays bounded and does not need this launcher.
"""
import os
import sys
import runpy

_HERE = os.path.dirname(os.path.abspath(__file__))   # .../reactemg/trahgr_lowmem
_REACTEMG = os.path.dirname(_HERE)                    # .../reactemg

# Run from reactemg/ so main.py's relative paths (../data, model_checkpoints/, output/,
# wandb/) resolve, and make both dirs importable.
os.chdir(_REACTEMG)
sys.path.insert(0, _HERE)
sys.path.insert(0, _REACTEMG)

# Import the stock modules that construct TraHGR datasets, then overwrite their
# TraHGR_Dataset reference with the lazy subclass. `preprocessing_utils.initialize_dataset`
# looks the name up in its own module globals at call time, so patching the module
# attribute is sufficient and takes effect for the training run.
import dataset as _dataset
import preprocessing_utils as _pp
from lowmem_dataset import TraHGR_Dataset as _LowMemTraHGR

_dataset.TraHGR_Dataset = _LowMemTraHGR
_pp.TraHGR_Dataset = _LowMemTraHGR

print("[trahgr_lowmem] Active: lazy low-RAM TraHGR_Dataset (identical outputs to stock).")

# Delegate to the stock entrypoint with the same CLI args (sys.argv[1:] is preserved).
runpy.run_path(os.path.join(_REACTEMG, "main.py"), run_name="__main__")
