"""
Memory-efficient (lazy) drop-in replacement for reactemg/dataset.py::TraHGR_Dataset.

WHY
---
The stock TraHGR_Dataset does EAGER preprocessing in __init__: it Butterworth-filters
every sliding window and materializes BOTH the temporal ([S, W*3]) and featural
([W/S, S*S*3]) patch tensors for EVERY window across ALL files, then holds them all in
RAM (~117 KB per window, float32). Because windows overlap heavily (stride 30 over a
600-sample window) and each is expanded ~6x by the 3-order filtering + dual patch
layout, large training pools (e.g. `pub_with_epn`) exhaust memory *during preprocessing*,
before training starts.

WHAT THIS DOES
--------------
This subclass keeps only the RAW per-file signal streams in memory (loaded once) plus a
lightweight (file_index, start_index) list, and computes the Butterworth filtering + patch
tensors on the fly in __getitem__ (parallelized across DataLoader workers). It reuses the
parent class's _load_file / _butterworth_filter / _create_patches verbatim, so every
returned sample is byte-for-byte identical to the eager version — only WHEN the work
happens and the peak memory differ.

Peak RAM drops from "all expanded patches for all windows" (tens of GB) to "the raw
dataset, stored once" (a couple of GB), with no change in numerical output.

ISOLATION
---------
This file imports the stock TraHGR_Dataset only to subclass it (read-only). It does not
modify any shared source, and it is activated only via trahgr_lowmem/run.py. No other
model variant is touched.
"""
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Stock class (eager). We reuse its _load_file / _butterworth_filter / _create_patches
# so the per-sample math is guaranteed identical.
from dataset import TraHGR_Dataset as _EagerTraHGR_Dataset


class TraHGR_Dataset(_EagerTraHGR_Dataset):
    """Lazy variant with the exact same constructor signature and __getitem__ contract."""

    def __init__(
        self,
        window_size: int,
        offset: int,
        file_paths,
        num_classes: int,
        butter_cutoff_hz: float = 90.0,
    ):
        # Intentionally DO NOT call the eager parent __init__ (that materializes every
        # window's patches). Initialize the torch Dataset and replicate ONLY the light
        # attribute setup that the inherited helper methods rely on.
        Dataset.__init__(self)
        self.window_size = int(window_size)
        self.offset = int(offset)
        self.file_paths = file_paths
        self.num_classes = int(num_classes)
        self.butter_cutoff_hz = float(butter_cutoff_hz)

        self.S = 8
        self.fs = 200
        self.C = 3
        self.W = self.window_size
        if self.W % self.S != 0:
            raise ValueError(
                f"Window size ({self.W}) must be divisible by number of sensors ({self.S})"
            )

        # Lazy storage: each file's padded stream stored ONCE + a flat window index.
        self._streams = []   # list of [T_pad, 8] float32   (raw, unfiltered)
        self._gts = []       # list of [T_pad]    int64
        self._index = []     # list of (file_idx, start_idx)

        for path in tqdm(self.file_paths, desc="[TraHGR low-mem] indexing"):
            data_array, action_sequence = self._load_file(path)  # inherited, identical
            if data_array.shape[0] < 100:
                continue

            # Left-pad by tiling the first 100 timesteps up to window_size (same as eager).
            pad_len = self.window_size
            seed = data_array[:100]
            seed_gt = action_sequence[:100]
            reps = int(np.ceil(pad_len / 100))
            pad_sig = np.tile(seed, (reps, 1))[:pad_len]
            pad_gt = np.tile(seed_gt, reps)[:pad_len]

            x = np.concatenate([pad_sig, data_array], axis=0).astype(np.float32)
            y = np.concatenate([pad_gt, action_sequence], axis=0).astype(np.int64)

            file_idx = len(self._streams)
            self._streams.append(x)
            self._gts.append(y)

            N = x.shape[0]
            for start_idx in range(0, N - self.window_size + 1, self.offset):
                self._index.append((file_idx, start_idx))

        if len(self._index) == 0:
            raise RuntimeError("No samples constructed")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        file_idx, start = self._index[idx]
        end = start + self.window_size
        window_emg = self._streams[file_idx][start:end, :]   # [W, 8]
        window_gt = self._gts[file_idx][start:end]           # [W]

        # Inherited, identical to the eager path.
        proc = self._butterworth_filter(window_emg)          # -> [S, W, 3]
        temporal, featural = self._create_patches(proc)      # -> [S, W*3], [W/S, S*S*3]

        return (
            torch.from_numpy(temporal),
            torch.from_numpy(featural),
            torch.tensor(int(window_gt[-1]), dtype=torch.long),
            torch.from_numpy(window_gt.copy()),
        )
