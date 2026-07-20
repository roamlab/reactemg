# trahgr_lowmem — memory-efficient TraHGR training

Self-contained fix for out-of-memory during **TraHGR training preprocessing**. Everything
TraHGR-specific lives in this folder; no stock source file is modified and no other model
variant (any2any, lstm, ann, ed_tcn, lda) is affected.

## The problem

`dataset.py::TraHGR_Dataset` preprocesses **eagerly** in `__init__`: it Butterworth-filters
every sliding window and stores both temporal (`[8, 1800]`) and featural (`[75, 192]`)
patch tensors for **every window across all files** in RAM (~117 KB/window). With heavy
window overlap (stride 30) and the ~6× filter/patch expansion, large pools like
`pub_with_epn` exhaust 64 GB before training starts.

## The fix

`lowmem_dataset.py` defines a `TraHGR_Dataset` subclass that stores only the **raw per-file
streams once** plus a lightweight `(file, start)` index, and computes the filtering + patches
**lazily in `__getitem__`** (parallelized by the DataLoader workers). It reuses the parent's
`_load_file` / `_butterworth_filter` / `_create_patches`, so **every sample is identical** —
only when the work happens and the peak memory differ. Peak RAM drops from tens of GB to a
couple of GB (the raw dataset stored once).

Only **training** needs this (it loads the whole pool at once). Evaluation builds the dataset
one file at a time, so it is already bounded.

## How to use

Run exactly like the batch-64 scripts, but from this folder:

```bash
# 1) train EPN 3-class, EPN 6-class, then pretrain (pub_with_epn)
bash trahgr_lowmem/train_trahgr_lowmem.sh

# 2) set CKPT_TRAHGR (line 15) to the pretrain checkpoint, then LOSO fine-tune on ROAM
bash trahgr_lowmem/finetune_trahgr_lowmem.sh
```

Or use the launcher directly with any `main.py` flags:

```bash
python3 trahgr_lowmem/run.py --model_choice trahgr --num_classes 3 \
  --dataset_selection pub_with_epn --window_size 600 --offset 30 \
  --epn_subset_percentage 1.0 --val_patient_ids s1 \
  --embedding_dim 144 --nhead 8 --num_layers 1 --batch_size 64 \
  --exp_name trahgr_pretrain_3class
```

`run.py` swaps in the lazy dataset in-process and delegates to the stock `main.py`. (As with
any training here, `wandb` must be importable — `pip install wandb`, and optionally
`export WANDB_MODE=disabled`.)

## Files

- `lowmem_dataset.py` — lazy `TraHGR_Dataset` subclass (identical outputs)
- `run.py` — drop-in launcher: patches in the lazy dataset, then runs stock `main.py`
- `train_trahgr_lowmem.sh` / `finetune_trahgr_lowmem.sh` — batch-64 run scripts routed through `run.py`
