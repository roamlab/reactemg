import torch
import random
import pandas as pd
import numpy as np
import scipy.signal
from torch.utils.data import Dataset
from scipy.signal import medfilt
from tqdm import tqdm
import math
from typing import List, Optional, Dict, Tuple
from torch import nn
import torch.nn.functional as F


############################################################
################################ ANN #######################
############################################################


class ANN_Dataset(Dataset):
    """
    1. Applies padding for the first window_size timesteps using the first 100 real timesteps,
    2. Slides windows of length 'window_size' with stride 'offset' (no partial windows),
    3. Extracts time-domain features (RMS, VAR, MAV, SSC, ZC, WL) per channel,
    4. Labels each window by the last timestep's label in that window,
    5. Stores the raw per-timestep ground truth of that window,
    6. Standardizes features, either by computing mean/std or by using precomputed stats.
    """

    def __init__(
        self,
        window_size,
        offset,
        file_paths,
        num_classes=10,
        use_precomputed_stats=False,
        precomputed_mean=None,
        precomputed_std=None,
    ):
        self.window_size = window_size
        self.offset = offset
        self.file_paths = file_paths
        self.num_classes = num_classes

        self.use_precomputed_stats = use_precomputed_stats
        self.precomputed_mean = precomputed_mean
        self.precomputed_std = precomputed_std

        self.all_features = []
        self.all_labels = []
        # Each element in self.all_raw_gt is a numpy array of shape [window_size]
        self.all_raw_gt = []

        # Temporary storage for unstandardized features
        tmp_feature_list = []

        # -------------------------------------------------------
        # Loop over each file using tqdm
        # -------------------------------------------------------
        for path in tqdm(self.file_paths, desc="Loading EMG files"):
            # 1) Load the file
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
                if "gt" not in df.columns:
                    raise Exception(f"'gt' column not found in {path}")

                action_sequence = df["gt"].to_numpy()
                try:
                    df_emg = df[
                        [
                            "emg_0",
                            "emg_1",
                            "emg_2",
                            "emg_3",
                            "emg_4",
                            "emg_5",
                            "emg_6",
                            "emg_7",
                        ]
                    ]
                except KeyError:
                    df_emg = df[
                        ["emg0", "emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7"]
                    ]
                data_array = df_emg.to_numpy()

            elif path.lower().endswith(".npy"):
                loaded = np.load(path).astype(np.float32)
                # First column is gt, rest are EMG channels
                action_sequence = loaded[:, 0]
                data_array = loaded[:, 1:]  # shape [N, 8]
            else:
                raise ValueError("File extension not recognized. Must be .csv or .npy")

            # data_array: shape [num_timesteps, 8]
            # action_sequence: shape [num_timesteps]

            # -------------------------------------------------------
            # 2) Construct "padding" of length window_size from first 100 timesteps.
            # -------------------------------------------------------
            if data_array.shape[0] < 100:
                print(f"Skipping {path}: only {data_array.shape[0]} timesteps (<100).")
                continue

            pad_section_emg = data_array[:100]
            pad_section_gt = action_sequence[:100]

            needed = self.window_size
            times_to_repeat = (needed // 100) + 1
            big_pad_emg = np.tile(pad_section_emg, (times_to_repeat, 1))[:needed]
            big_pad_gt = np.tile(pad_section_gt, (times_to_repeat))[:needed]

            # Concatenate pad + actual data
            padded_emg = np.concatenate([big_pad_emg, data_array], axis=0)
            padded_gt = np.concatenate([big_pad_gt, action_sequence], axis=0)

            total_padded_length = padded_emg.shape[0]

            # -------------------------------------------------------
            # 3) Sliding windows in a for loop
            #    This ensures no zero-sized or partial windows
            # -------------------------------------------------------
            max_start = total_padded_length - self.window_size

            if max_start < 0:
                # Not enough data to form even a single window
                continue

            for start_idx in range(0, max_start + 1, self.offset):
                end_idx = start_idx + self.window_size
                window_emg = padded_emg[start_idx:end_idx, :]  # shape [window_size, 8]
                window_gt = padded_gt[start_idx:end_idx]
                label_for_window = window_gt[-1]  # last timestep's label

                # 4) Extract time-domain features
                feats_1d = self._compute_window_features(window_emg)
                tmp_feature_list.append(feats_1d)

                self.all_labels.append(label_for_window)
                self.all_raw_gt.append(window_gt)

        self.all_features = np.array(
            tmp_feature_list, dtype=np.float32
        )  # shape [N_windows, 48]
        self.all_labels = np.array(self.all_labels, dtype=np.int64)
        self.all_raw_gt = np.array(self.all_raw_gt, dtype=np.int64)

        # -------------------------------------------------------
        # 5) Standardization
        # -------------------------------------------------------
        if self.use_precomputed_stats:
            if (self.precomputed_mean is None) or (self.precomputed_std is None):
                raise ValueError(
                    "use_precomputed_stats=True requires both precomputed_mean and precomputed_std."
                )
            self.mean_ = self.precomputed_mean
            self.std_ = self.precomputed_std
        else:
            # Compute our own mean & std from the entire dataset
            self.mean_ = self.all_features.mean(axis=0, keepdims=True)  # [1, 48]
            self.std_ = self.all_features.std(axis=0, keepdims=True)  # [1, 48]

        eps = 1e-8
        self.std_[self.std_ < eps] = eps

        self.all_features = (self.all_features - self.mean_) / self.std_

    def __len__(self):
        return len(self.all_features)

    def __getitem__(self, idx):
        """
        Return (X, y, raw_gt_seq) where
          X : shape [48], the standardized features
          y : scalar int label
          raw_gt_seq : np.array([window_size,]) of int ground-truth labels
        """
        X = torch.FloatTensor(self.all_features[idx])
        y = torch.tensor(self.all_labels[idx], dtype=torch.long)
        raw_gt_seq = torch.LongTensor(self.all_raw_gt[idx])

        return X, y, raw_gt_seq

    @staticmethod
    def _compute_window_features(window_emg):
        """
        Compute the 6 time-domain features for each channel in the window.
        window_emg shape: [window_size, 8].
        Returns a 1D array of shape [48].
        Features in order: [RMS, VAR, MAV, SSC, ZC, WL].
        """
        T = window_emg.shape[0]

        def zero_crossing(channel_data):
            zc_count = 0
            for i in range(T - 1):
                if (channel_data[i] * channel_data[i + 1]) < 0:
                    zc_count += 1
            return zc_count

        def slope_sign_change(channel_data):
            ssc_count = 0
            for i in range(1, T - 1):
                prev_diff = channel_data[i] - channel_data[i - 1]
                next_diff = channel_data[i] - channel_data[i + 1]
                if prev_diff * next_diff < 0:
                    ssc_count += 1
            return ssc_count

        def waveform_length(channel_data):
            wl = 0
            for i in range(T - 1):
                wl += abs(channel_data[i + 1] - channel_data[i])
            return wl

        features_out = []
        for ch in range(window_emg.shape[1]):
            channel_data = window_emg[:, ch]
            rms_val = np.sqrt(np.mean(channel_data**2))
            var_val = np.var(channel_data)
            mav_val = np.mean(np.abs(channel_data))
            ssc_val = slope_sign_change(channel_data)
            zc_val = zero_crossing(channel_data)
            wl_val = waveform_length(channel_data)
            features_out.extend([rms_val, var_val, mav_val, ssc_val, zc_val, wl_val])

        return np.array(features_out, dtype=np.float32)



############################################################
########################## TraHGR ##########################
############################################################
def scale_to_unit(x):
    return np.clip(x / 128.0, -1.0, 1.0)

class TraHGR_Dataset(Dataset):
    def __init__(
        self,
        window_size: int,
        offset: int,
        file_paths: List[str],
        num_classes: int,
        butter_cutoff_hz: float = 90.0,
    ):
        super().__init__()
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
            raise ValueError(f"Window size ({self.W}) must be divisible by number of sensors ({self.S})")

        self.all_temporal = []
        self.all_featural = []
        self.all_labels = []
        self.all_raw_gt = []

        # Loop over each file
        for path in tqdm(self.file_paths):
            data_array, action_sequence = self._load_file(path)
            if data_array.shape[0] < 100:
                continue

            # Create padding from first 100 timesteps
            pad_len = self.window_size
            seed = data_array[:100]
            seed_gt = action_sequence[:100]
            reps = int(np.ceil(pad_len / 100))
            pad_sig = np.tile(seed, (reps, 1))[:pad_len]
            pad_gt = np.tile(seed_gt, reps)[:pad_len]

            # Concatenate padding with actual data
            x = np.concatenate([pad_sig, data_array], axis=0)
            y = np.concatenate([pad_gt, action_sequence], axis=0)
            N = x.shape[0]

            # Sliding windows
            for start_idx in range(0, N - self.window_size + 1, self.offset):
                end_idx = start_idx + self.window_size
                window_emg = x[start_idx:end_idx, :]
                window_gt = y[start_idx:end_idx]

                label_for_window = int(window_gt[-1])

                # Apply TraHGR preprocessing
                proc = self._butterworth_filter(window_emg)
                temporal, featural = self._create_patches(proc)

                self.all_temporal.append(temporal)
                self.all_featural.append(featural)
                self.all_labels.append(label_for_window)
                self.all_raw_gt.append(window_gt.copy())


        if len(self.all_labels) == 0:
            raise RuntimeError("No samples constructed")

        self.all_temporal = np.asarray(self.all_temporal, dtype=np.float32)
        self.all_featural = np.asarray(self.all_featural, dtype=np.float32)
        self.all_labels = np.asarray(self.all_labels, dtype=np.int64)
        self.all_raw_gt = np.asarray(self.all_raw_gt, dtype=np.int64)

    def _load_file(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        if path.lower().endswith('.csv'):
            df = pd.read_csv(path)
            if 'gt' not in df.columns:
                raise Exception('gt column not found')
            action_sequence = df['gt'].to_numpy()
            try:
                df_emg = df[[
                    'emg_0','emg_1','emg_2','emg_3','emg_4','emg_5','emg_6','emg_7'
                ]]
            except KeyError:
                df_emg = df[[
                    'emg0','emg1','emg2','emg3','emg4','emg5','emg6','emg7'
                ]]
            data_array = df_emg.to_numpy().astype(np.float32)
        elif path.lower().endswith('.npy'):
            loaded = np.load(path).astype(np.float32)
            action_sequence = loaded[:, 0]
            data_array = loaded[:, 1:]
        else:
            raise ValueError("File extension not recognized. Must be .csv or .npy")
        return data_array, action_sequence

    def _butterworth_filter(self, window_emg: np.ndarray) -> np.ndarray:
        """Apply Butterworth filtering with orders 1,3,5"""
        from scipy.signal import butter, filtfilt

        W, S = window_emg.shape
        assert S == self.S, f"Expected S={self.S}, got {S}"

        proc = np.zeros((self.S, self.W, self.C), dtype=np.float32)
        orders = [1, 3, 5]

        for s in range(self.S):
            # Scale to [-1, 1] range
            sig_raw = window_emg[:, s].astype(np.float32)
            sig = scale_to_unit(sig_raw)

            # Apply filtering at each order
            for ci, order in enumerate(orders):
                b, a = butter(order, self.butter_cutoff_hz, btype='low', fs=self.fs)
                filt = filtfilt(b, a, sig).astype(np.float32)
                proc[s, :, ci] = filt

        return proc

    def _create_patches(self, processed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create temporal and featural patches"""
        S, W, C = processed.shape
        assert S == self.S and W == self.W and C == self.C

        # Temporal patches: one token per sensor
        temporal = processed.reshape(S, W * C)

        # Featural patches: non-overlapping blocks
        Nf = W // S
        featural = np.zeros((Nf, S * S * C), dtype=np.float32)
        for i in range(Nf):
            t0 = i * S
            t1 = t0 + S
            block = processed[:, t0:t1, :]
            featural[i] = block.reshape(-1)
        return temporal.astype(np.float32), featural.astype(np.float32)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx: int):
        temporal = torch.from_numpy(self.all_temporal[idx])
        featural = torch.from_numpy(self.all_featural[idx])
        y = torch.tensor(self.all_labels[idx], dtype=torch.long)
        raw_gt = torch.from_numpy(self.all_raw_gt[idx])
        return temporal, featural, y, raw_gt


##########################################################
########################## LSTM ##########################
##########################################################
class LSTM_Dataset(Dataset):
    """
    1. Takes an outer sliding window of size `window_size` with stride `offset`.
    2. For each outer window, scans it with an inner window of size 100 (stride=1).
    3. Computes std dev across each channel => (seq_len, num_channels).
    4. Assigns labels by the mode in that inner window => (seq_len,).
    5. Normalizes the resulting features with either user-supplied mean/std or newly computed from this dataset.
    """

    def __init__(
        self,
        window_size,
        offset,
        csv_paths,
        num_classes,
        precomputed_mean=None,
        precomputed_std=None,
    ):
        super().__init__()
        self.window_size = window_size
        self.offset = offset
        self.csv_paths = csv_paths
        self.num_classes = num_classes

        self.precomputed_mean = precomputed_mean
        self.precomputed_std = precomputed_std

        self.samples = []
        self.global_mean = None
        self.global_std = None
        self._raw_features_accumulator = []

        for path in tqdm(self.csv_paths, desc="Loading data"):
            data_array, action_sequence = self._load_file(path)
            N = data_array.shape[0]

            for i in range(0, N - self.window_size + 1, self.offset):
                seg_data = data_array[i : i + self.window_size]
                seg_label = action_sequence[i : i + self.window_size]

                seq_length = self.window_size - 100
                X = np.zeros((seq_length, 8), dtype=np.float32)
                Y = np.zeros((seq_length,), dtype=np.int64)

                for t in range(seq_length):
                    inner_win_data = seg_data[t : t + 100]
                    inner_win_label = seg_label[t : t + 100]
                    std_vals = np.std(inner_win_data, axis=0, dtype=np.float32)
                    X[t] = std_vals

                    label_counts = np.bincount(inner_win_label.astype(int))
                    mode_label = np.argmax(label_counts)
                    Y[t] = mode_label

                if self.precomputed_mean is None or self.precomputed_std is None:
                    self._raw_features_accumulator.append(X)

                self.samples.append((X, Y, seg_label))

        if (self.precomputed_mean is not None) and (self.precomputed_std is not None):
            self.global_mean = self.precomputed_mean
            self.global_std = self.precomputed_std
        else:
            if len(self._raw_features_accumulator) == 0:
                raise RuntimeError("No data found to compute mean/std.")
            all_features_stacked = np.concatenate(
                self._raw_features_accumulator, axis=0
            )
            self.global_mean = np.mean(all_features_stacked, axis=0, dtype=np.float32)
            self.global_std = np.std(all_features_stacked, axis=0, dtype=np.float32)

        for idx, (X, Y, seg_label) in enumerate(self.samples):
            X_normed = (X - self.global_mean) / (self.global_std + 1e-8)
            self.samples[idx] = (X_normed, Y, seg_label)

        self._raw_features_accumulator = None

    def _load_file(self, path):
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
            if "gt" not in df.columns:
                raise Exception("gt column not found in CSV.")
            action_sequence = df["gt"].to_numpy()
            try:
                df_emg = df[
                    [
                        "emg_0",
                        "emg_1",
                        "emg_2",
                        "emg_3",
                        "emg_4",
                        "emg_5",
                        "emg_6",
                        "emg_7",
                    ]
                ]
            except KeyError:
                df_emg = df[
                    ["emg0", "emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7"]
                ]
            data_array = df_emg.to_numpy()
        elif path.lower().endswith(".npy"):
            loaded = np.load(path).astype(np.float32)
            action_sequence = loaded[:, 0]
            data_array = loaded[:, 1:]
        else:
            raise ValueError("File extension not recognized. Must be .csv or .npy")

        return data_array, action_sequence

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, Y, seg_label = self.samples[idx]
        X_t = torch.FloatTensor(X)
        Y_t = torch.LongTensor(Y)
        raw_gt_t = torch.LongTensor(seg_label)

        return X_t, Y_t, raw_gt_t


############################################################
########################## ED-TCN ##########################
############################################################
class EDTCN_Dataset(Dataset):
    """
    1. Outer window of size `window_size` with stride `offset`.
    2. Inner window of size 150 with stride 25 (default) to produce 19 subwindows,
       each subwindow computing MAV across 8 channels. The label for each subwindow is
       taken from the last raw timestep in that subwindow.
    """

    def __init__(
        self, window_size, offset, file_paths, inner_window_size=150, inner_stride=25
    ):
        super().__init__()
        self.window_size = window_size
        self.offset = offset
        self.file_paths = file_paths
        self.inner_window_size = inner_window_size
        self.inner_stride = inner_stride

        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        for path in tqdm(self.file_paths):
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
                if "gt" not in df.columns:
                    raise Exception("gt column not found in CSV.")
                action_sequence = df["gt"].to_numpy()
                try:
                    df_emg = df[
                        [
                            "emg_0",
                            "emg_1",
                            "emg_2",
                            "emg_3",
                            "emg_4",
                            "emg_5",
                            "emg_6",
                            "emg_7",
                        ]
                    ]
                except KeyError:
                    df_emg = df[
                        ["emg0", "emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7"]
                    ]
                data_array = df_emg.to_numpy()
            elif path.lower().endswith(".npy"):
                loaded = np.load(path).astype(np.float32)
                action_sequence = loaded[:, 0]
                data_array = loaded[:, 1:]
            else:
                raise ValueError("File extension not recognized. Must be .csv or .npy")

            N = data_array.shape[0]
            start_idx = 0
            while (start_idx + self.window_size) <= N:
                end_idx = start_idx + self.window_size
                window_emg = data_array[start_idx:end_idx]
                window_labels = action_sequence[start_idx:end_idx]

                mav_seq, label_seq = self._compute_mav_sequence(
                    window_emg, window_labels
                )

                self.samples.append((mav_seq, label_seq, window_labels))
                start_idx += self.offset

    def _compute_mav_sequence(self, window_emg, window_labels):
        w_size = window_emg.shape[0]
        iwin = self.inner_window_size
        istr = self.inner_stride

        mav_list = []
        label_list = []
        i_start = 0
        while (i_start + iwin) <= w_size:
            i_end = i_start + iwin
            sub_emg = window_emg[i_start:i_end, :]
            channel_mav = np.mean(np.abs(sub_emg), axis=0)
            sub_label = window_labels[i_end - 1]
            mav_list.append(channel_mav)
            label_list.append(sub_label)
            i_start += istr

        mav_seq = np.stack(mav_list, axis=0)
        label_seq = np.array(label_list, dtype=int)
        return mav_seq, label_seq

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mav_seq, label_seq, raw_gt = self.samples[idx]
        X = torch.FloatTensor(mav_seq)
        Y = torch.LongTensor(label_seq)
        raw_gt_t = torch.LongTensor(raw_gt)

        return X, Y, raw_gt_t


#################################################################
######################## Any2Any Dataset ########################
#################################################################
class Any2Any_Dataset(Dataset):
    def __init__(
        self,
        labeled_csv_paths,
        unlabeled_csv_paths,
        median_filter_size,
        window_size,
        offset,
        embedding_method,
        lambda_poisson,
        seeded_mask,
        sampling_probability_poisson,
        poisson_mask_percentage_sampling_range,
        end_mask_percentage_sampling_range,
        task_selection,
        stage_1_weights,
        stage_2_weights,
        mask_alignment,
        transition_buffer,
        mask_tokens_dict,
        with_training_curriculum,
        num_classes,
        medfilt_order,
        noise,
        hand_choice,
        inner_window_size,
        use_mav_for_emg,
        eval_mode=False,
        eval_task=None,
        transition_samples_only=False,
        mask_percentage=0.6,
        mask_type="poisson",
    ):
        """
        This dataset supports both labeled and unlabeled data in csv/numpy format, applies
        median filtering and rectification, optionally performs ED-TCN-style MAV extraction,
        and prepares data for various tasks (predict action, predict EMG, etc.) with masking.

        The dataset can operate in training mode (with curriculum learning over multiple stages)
        or evaluation mode (with a fixed masking scheme). It also handles transition samples
        and coarse/fine segmentation of the action labels.

        This dataset is used for both training and simulated online inference

        Args:
            labeled_csv_paths (List[str]):
                Paths to labeled CSV/NPY files containing EMG data and action labels.
            unlabeled_csv_paths (List[str]):
                Paths to unlabeled CSV/NPY files containing EMG data.
            median_filter_size (int):
                Kernel size for median filtering of EMG signals.
            window_size (int):
                Number of timesteps per sample (outer window size).
            offset (int):
                Step size to slide over the sequence when creating samples.
            embedding_method (str):
                Method for representing EMG/action tokens (e.g. "linear_projection").
            lambda_poisson (float):
                Lambda parameter for Poisson-based masking.
            seeded_mask (bool):
                If True, uses a seeded (deterministic) approach to generate masks.
            sampling_probability_poisson (float):
                Probability of selecting Poisson masking vs. other types (e.g., "end").
            poisson_mask_percentage_sampling_range (Dict[int, Tuple[float, float]]):
                Per-task range for sampling the percentage of Poisson-mask coverage.
            end_mask_percentage_sampling_range (Dict[int, Tuple[float, float]]):
                Per-task range for sampling the percentage of end-based mask coverage.
            task_selection (List[int]):
                List of task indices. Tasks map to:
                0 = Predict action from EMG (dense labeling),
                1 = Predict EMG from action,
                2 = Bidirectional (mask both EMG and action),
                3 = Unlabeled/self-supervised EMG.
            stage_1_weights (List[float]):
                Weights used for sampling stage 0 vs. stage 1 in the curriculum.
            stage_2_weights (List[float]):
                Weights used for sampling stage 0 vs. stage 2 in the curriculum.
            mask_alignment (str):
                Mask alignment strategy: "aligned" or "non-aligned" across channels.
            transition_buffer (int):
                Number of timesteps around the transition index for targeted masking.
            mask_tokens_dict (Dict[str, Dict[str, float]]):
                Dictionary specifying mask token values per embedding method (e.g. for EMG, action).
            with_training_curriculum (bool):
                If True, uses a curriculum learning approach over multiple stages.
            num_classes (int):
                Number of action classes (used for classification).
            medfilt_order (str):
                Whether median filtering is done before or after rectification ("before_rec" or "after_rec").
            noise (float):
                Amplitude of uniform noise added to EMG signals (0.0 for no noise).
            hand_choice (str):
                Which hand's EMG data is being used, "left" or "right" (remaps channels for "left").
            inner_window_size (int):
                Size of subwindows for coarse labeling or ED-TCN MAV extraction.
            use_mav_for_emg (int):
                If 1, performs MAV extraction in an inner subwindow loop (ED-TCN style).
            eval_mode (bool, optional):
                If True, dataset is used for evaluation (fixed masking). Defaults to False.
            eval_task (str, optional):
                Task for evaluation mode. One of {"predict_action", "predict_emg", "predict_emg_ss"}.
                Defaults to None.
            transition_samples_only (bool, optional):
                If True, filters labeled data to only keep samples containing transitions.
                Defaults to False.
            mask_percentage (float, optional):
                Percentage of time steps to mask during evaluation. Defaults to 0.6.
            mask_type (str, optional):
                Masking strategy for evaluation. One of {"poisson", "end", "targeted"}.
                Defaults to "poisson".

        Attributes:
            all_data (List[Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]]):
                Master list of all loaded samples (EMG, action, transition_index, coarse_label).
            raw_labeled_data (List[Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]]):
                Loaded labeled samples (before augmentation/masking).
            noisy_labeled_data (List[Any]):
                Not currently used (placeholder for future expansions).
            augmented_transition_samples (List[Any]):
                Placeholder list for additional transition-augmented samples.
            raw_unlabeled_data (List[Any]):
                Loaded unlabeled samples (placeholder for future expansions).
            noisy_unlabeled_data (List[Any]):
                Placeholder list for unlabeled noisy samples.
            untokenized_data (List[Tuple[np.ndarray, np.ndarray, int, Optional[np.ndarray]]]):
                Stores unmasked data for evaluation/analysis.

        Raises:
            Exception: If the CSV does not contain a "gt" column for action labels.
            ValueError: If file extension is not recognized, or if `medfilt_order`
                or `mask_type` is invalid.
        """

        self.all_data = []
        self.raw_labeled_data = []
        self.noisy_labeled_data = []
        self.augmented_transition_samples = []
        self.raw_unlabeled_data = []
        self.noisy_unlabeled_data = []
        self.window_size = window_size
        self.embedding_method = embedding_method
        self.lambda_poisson = lambda_poisson
        self.sampling_probability_poisson = sampling_probability_poisson
        self.poisson_mask_percentage_sampling_range = (
            poisson_mask_percentage_sampling_range
        )
        self.end_mask_percentage_sampling_range = end_mask_percentage_sampling_range
        self.seeded_mask = seeded_mask
        self.task_selection = task_selection
        self.mask_alignment = mask_alignment
        self.transition_buffer = transition_buffer
        self.mask_tokens_dict = mask_tokens_dict
        self.num_classes = num_classes
        self.noise_amplitude = noise
        self.hand_choice = hand_choice
        self.inner_window_size = inner_window_size
        self.use_mav_for_emg = use_mav_for_emg

        self.curriculum_stage = 0
        self.stage_1_weights = stage_1_weights
        self.stage_2_weights = stage_2_weights
        self.with_training_curriculum = with_training_curriculum
        self.class_data = {0: 0, 1: 0, 2: 0}

        self.cur_epoch = 0

        self.eval_mode = eval_mode
        self.eval_task = eval_task
        self.transition_samples_only = transition_samples_only
        self.eval_mask_percentage = mask_percentage
        self.eval_mask_type = mask_type
        self.untokenized_data = []

        if not self.eval_mode:
            if any(item in [0, 1, 2] for item in self.task_selection):
                for path in tqdm(labeled_csv_paths):
                    temp_raw_labeled_data, temp_noisy_labeled_data = (
                        self.multivariate_preprocessing(
                            path,
                            median_filter_size,
                            window_size,
                            offset,
                            embedding_method,
                            medfilt_order,
                            hand_choice,
                        )
                    )
                    self.raw_labeled_data.extend(temp_raw_labeled_data)

                self.all_data.extend(self.raw_labeled_data)
                self.all_data.extend(self.noisy_labeled_data)
                self.all_data.extend(self.augmented_transition_samples)

            self.modality_switch_index = len(self.all_data)

            if any(item in [3] for item in self.task_selection):
                for path in tqdm(unlabeled_csv_paths):
                    temp_raw_unlabeled_data, temp_noisy_unlabeled_data = (
                        self.multivariate_preprocessing(
                            path,
                            median_filter_size,
                            window_size,
                            offset,
                            embedding_method,
                            medfilt_order,
                            hand_choice,
                        )
                    )
                    self.all_data.extend(temp_raw_unlabeled_data)
                    self.all_data.extend(temp_noisy_unlabeled_data)

            if 3 not in self.task_selection:
                self.len_multiplier = len(self.task_selection)
            else:
                self.len_multiplier = len(self.task_selection) - 1
            self.num_labeled_samples = self.modality_switch_index * self.len_multiplier
            self.num_unlabeled_samples = len(self.all_data) - self.modality_switch_index
            self.labeled_indices = list(range(self.num_labeled_samples))
            self.unlabeled_indices = list(
                range(
                    self.num_labeled_samples,
                    self.num_labeled_samples + self.num_unlabeled_samples,
                )
            )

        else:
            for path in tqdm(labeled_csv_paths):
                temp_raw_labeled_data, _ = self.multivariate_preprocessing(
                    path,
                    median_filter_size,
                    window_size,
                    offset,
                    embedding_method,
                    medfilt_order,
                    hand_choice,
                )
                temp_raw_labeled_data_untokenized, _ = self.multivariate_preprocessing(
                    path,
                    median_filter_size,
                    window_size,
                    offset,
                    embedding_method,
                    medfilt_order,
                    hand_choice,
                )

                if self.transition_samples_only:
                    filtered_data = []
                    filtered_untok = []
                    for sample_idx in range(len(temp_raw_labeled_data)):
                        (
                            _emg,
                            _act,
                            _tindex,
                            _coarse_action,
                        ) = temp_raw_labeled_data[sample_idx]
                        if _tindex != -1:
                            filtered_data.append(temp_raw_labeled_data[sample_idx])
                            filtered_untok.append(
                                temp_raw_labeled_data_untokenized[sample_idx]
                            )
                    temp_raw_labeled_data = filtered_data
                    temp_raw_labeled_data_untokenized = filtered_untok

                self.all_data.extend(temp_raw_labeled_data)
                self.untokenized_data.extend(temp_raw_labeled_data_untokenized)

            self.modality_switch_index = 0
            self.num_labeled_samples = len(self.all_data)
            self.num_unlabeled_samples = 0
            self.labeled_indices = list(range(self.num_labeled_samples))
            self.unlabeled_indices = []

    # (use_mav_for_emg): helper to do ED-TCN-style MAV
    def _extract_mav_sequence(
        self, raw_emg_window, raw_action_window, inner_window_size, inner_stride
    ):
        """
        Extracts MAV with the same logic as ED-TCN:
          - Slide an inner window of size `inner_window_size` (e.g. 150)
            with stride `inner_stride` (e.g. 25).
          - For each subwindow, compute mean(abs(emg)) across 8 channels => shape (8,).
          - Label is the last raw action in that subwindow.
          - Return (mav_sequence, action_sequence) each with length = #subwindows.
        """
        T = raw_emg_window.shape[0]
        i_start = 0
        mav_list = []
        act_list = []
        while (i_start + inner_window_size) <= T:
            i_end = i_start + inner_window_size
            sub_emg = raw_emg_window[i_start:i_end]
            channel_mav = np.mean(np.abs(sub_emg), axis=0)
            label_sub = raw_action_window[i_end - 1]
            mav_list.append(channel_mav)
            act_list.append(label_sub)
            i_start += inner_stride

        if len(mav_list) == 0:
            return None, None
        mav_seq = np.stack(mav_list, axis=0)  # (num_subwindows, 8)
        act_seq = np.array(act_list, dtype=int)  # (num_subwindows,)
        return mav_seq, act_seq

    def multivariate_preprocessing(
        self,
        path,
        median_filter_size,
        window_size,
        offset,
        embedding_method,
        medfilt_order,
        hand_choice,
    ):
        extracted_samples = []
        extracted_unlabeled_samples = []

        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
            if "gt" not in df.columns:
                raise Exception("gt column not found")
            action_sequence = df["gt"].to_numpy()
            try:
                df_emg = df[
                    [
                        "emg_0",
                        "emg_1",
                        "emg_2",
                        "emg_3",
                        "emg_4",
                        "emg_5",
                        "emg_6",
                        "emg_7",
                    ]
                ]
            except KeyError:
                df_emg = df[
                    ["emg0", "emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7"]
                ]
            if hand_choice == "left":
                remap_order = [6, 5, 4, 3, 2, 1, 0, 7]
                data_array = df_emg.to_numpy().astype(np.int16)
                data_array = data_array[:, remap_order]
                df_emg = pd.DataFrame(
                    data_array, columns=["emg_" + str(i) for i in range(8)]
                )

            if medfilt_order == "before_rec":
                filtered_data = df_emg.apply(
                    lambda x: medfilt(x, kernel_size=median_filter_size)
                )
                rectified_data = np.abs(filtered_data)
            elif medfilt_order == "after_rec":
                rectified_data = np.abs(df_emg)
                rectified_data = rectified_data.apply(
                    lambda x: medfilt(x, kernel_size=median_filter_size)
                )
            else:
                raise ValueError(
                    "medfilt_order must be either 'before_rec' or 'after_rec'"
                )
            scaled_data = rectified_data / 128.0

        elif path.lower().endswith(".npy"):
            loaded = np.load(path).astype(np.float32)
            action_sequence = loaded[:, 0]
            data_array = loaded[:, 1:]
            if hand_choice == "left":
                remap_order = [6, 5, 4, 3, 2, 1, 0, 7]
                data_array = data_array[:, remap_order]

            if medfilt_order == "before_rec":
                for i in range(data_array.shape[1]):
                    data_array[:, i] = medfilt(
                        data_array[:, i], kernel_size=median_filter_size
                    )
                data_array = np.abs(data_array)
            elif medfilt_order == "after_rec":
                data_array = np.abs(data_array)
                for i in range(data_array.shape[1]):
                    data_array[:, i] = medfilt(
                        data_array[:, i], kernel_size=median_filter_size
                    )
            else:
                raise ValueError(
                    "medfilt_order must be either 'before_rec' or 'after_rec'"
                )
            scaled_data = data_array / 128.0
        else:
            raise ValueError("File extension not recognized. Must be .csv or .npy")

        if isinstance(scaled_data, pd.DataFrame):
            clipped_data = scaled_data.to_numpy().astype(np.float32)
        else:
            clipped_data = scaled_data.astype(np.float32)

        # NEW (use_mav_for_emg): If we want to do ED-TCN-style MAV extraction,
        # we create subwindows for each outer window. Then each sample is smaller in time dimension.
        if self.embedding_method == "linear_projection" and self.use_mav_for_emg == 1:
            # We do an outer sliding window of size window_size & offset, then inside that window we do MAV subwindows.
            N = clipped_data.shape[0]
            for start in range(0, N - window_size + 1, offset):
                raw_emg_window = clipped_data[start : start + window_size, :]
                raw_act_window = action_sequence[start : start + window_size]

                # call the MAV function:
                mav_seq, act_seq = self._extract_mav_sequence(
                    raw_emg_window,
                    raw_act_window,
                    self.inner_window_size,  # interpret as "inner_window_size" for MAV
                    getattr(self, "mav_inner_stride", 25),
                )
                if mav_seq is None:
                    continue

                # Now find the transition index in the "act_seq"
                transition_list = np.where(act_seq[:-1] != act_seq[1:])[0]
                transition_index = (
                    transition_list[0] if len(transition_list) > 0 else -1
                )

                # We'll not do additional sub-subwindows for "coarse" here,
                # but store the result as if (window_size == len(mav_seq)).
                # Coarse action is None, to keep consistent shape in the rest of the code.
                extracted_samples.append(
                    (
                        mav_seq.astype(np.float32),  # shape (#subwindows, 8)
                        act_seq.astype(np.int64),  # shape (#subwindows,)
                        transition_index,
                        None,
                    )
                )

        else:
            # main approach (no MAV, either raw or normal coarse):
            for start in range(0, clipped_data.shape[0] - window_size + 1, offset):
                window = clipped_data[start : start + window_size, :]
                windowed_action_sequence = action_sequence[start : start + window_size]

                transition_list = np.where(
                    windowed_action_sequence[:-1] != windowed_action_sequence[1:]
                )[0]
                transition_index = (
                    transition_list[0] if len(transition_list) > 0 else -1
                )

                if self.window_size > self.inner_window_size:
                    coarse_length = self.window_size // self.inner_window_size
                    coarse_action = np.zeros((coarse_length,), dtype=np.int64)
                    for i in range(coarse_length):
                        end_t = (i + 1) * self.inner_window_size
                        coarse_action[i] = windowed_action_sequence[end_t - 1]
                    extracted_samples.append(
                        (
                            window.astype(np.float32),
                            windowed_action_sequence.astype(np.int64),
                            transition_index,
                            coarse_action,
                        )
                    )
                else:
                    extracted_samples.append(
                        (
                            window.astype(np.float32),
                            windowed_action_sequence.astype(np.int64),
                            transition_index,
                            None,
                        )
                    )

        return extracted_samples, extracted_unlabeled_samples

    def masking_a2a(
        self,
        sequence,
        mask_percentage,
        mask_type,
        mask_channel_selection,
        mask_alignment,
        lambda_poisson,
        seeded_mask,
        mask_token,
        transition_index,
        transition_buffer,
        use_bert_mask,
    ):
        univariate_sequence_status = False
        if sequence.ndim == 1:
            univariate_sequence_status = True
            sequence = sequence.reshape(-1, 1)

        window_size, num_channel = sequence.shape

        if mask_type not in ["poisson", "end", "targeted"]:
            raise ValueError("mask_type must be 'poisson', 'end', or 'targeted'")

        if any(channel >= num_channel for channel in mask_channel_selection):
            raise ValueError("mask_channel_selection contains invalid channel indices")

        total_tokens_to_mask = int(window_size * mask_percentage)
        mask_positions = np.zeros_like(sequence, dtype=bool)
        rng = np.random if seeded_mask else np.random.default_rng()

        if mask_type == "poisson":
            if mask_alignment == "aligned":
                tokens_masked = 0
                while tokens_masked < total_tokens_to_mask:
                    span_length = rng.poisson(lambda_poisson)
                    if span_length <= 0 or span_length >= window_size:
                        continue
                    if tokens_masked + span_length > total_tokens_to_mask:
                        span_length = total_tokens_to_mask - tokens_masked
                    start_pos = (
                        rng.randint(0, max(1, window_size - span_length + 1))
                        if seeded_mask
                        else rng.integers(0, max(1, window_size - span_length + 1))
                    )
                    end_pos = start_pos + span_length
                    already_masked = mask_positions[
                        start_pos:end_pos, mask_channel_selection[0]
                    ]
                    positions_not_already_masked_indices = np.where(~already_masked)[0]
                    num_new_positions = len(positions_not_already_masked_indices)
                    if num_new_positions == 0:
                        continue
                    if tokens_masked + num_new_positions > total_tokens_to_mask:
                        num_needed = total_tokens_to_mask - tokens_masked
                        positions_not_already_masked_indices = (
                            positions_not_already_masked_indices[:num_needed]
                        )
                        num_new_positions = len(positions_not_already_masked_indices)
                    for channel in mask_channel_selection:
                        mask_positions[start_pos:end_pos, channel][
                            positions_not_already_masked_indices
                        ] = True
                    tokens_masked += num_new_positions

            elif mask_alignment == "non-aligned":
                for channel in mask_channel_selection:
                    tokens_masked = 0
                    while tokens_masked < total_tokens_to_mask:
                        span_length = rng.poisson(lambda_poisson)
                        if span_length <= 0 or span_length >= window_size:
                            continue
                        if tokens_masked + span_length > total_tokens_to_mask:
                            span_length = total_tokens_to_mask - tokens_masked
                        start_pos = (
                            rng.randint(0, max(1, window_size - span_length + 1))
                            if seeded_mask
                            else rng.integers(0, max(1, window_size - span_length + 1))
                        )
                        end_pos = start_pos + span_length
                        already_masked = mask_positions[start_pos:end_pos, channel]
                        positions_not_already_masked_indices = np.where(
                            ~already_masked
                        )[0]
                        num_new_positions = len(positions_not_already_masked_indices)
                        if num_new_positions == 0:
                            continue
                        if tokens_masked + num_new_positions > total_tokens_to_mask:
                            num_needed = total_tokens_to_mask - tokens_masked
                            positions_not_already_masked_indices = (
                                positions_not_already_masked_indices[:num_needed]
                            )
                            num_new_positions = len(
                                positions_not_already_masked_indices
                            )
                        mask_positions[start_pos:end_pos, channel][
                            positions_not_already_masked_indices
                        ] = True
                        tokens_masked += num_new_positions

        elif mask_type == "end":
            start_pos = window_size - total_tokens_to_mask
            for channel in mask_channel_selection:
                mask_positions[start_pos:, channel] = True

        elif mask_type == "targeted":
            for channel in mask_channel_selection:
                mask_positions[
                    transition_index - transition_buffer : transition_index
                    + transition_buffer,
                    channel,
                ] = True

        masked_sequence = np.where(mask_positions, mask_token, sequence)
        if univariate_sequence_status:
            masked_sequence = masked_sequence.squeeze()
            mask_positions = mask_positions.squeeze()
        return masked_sequence, mask_positions

    def __len__(self):
        if not self.eval_mode:
            if self.num_unlabeled_samples == 0:
                return self.num_labeled_samples
            else:
                return self.num_labeled_samples + self.num_unlabeled_samples
        else:
            return len(self.all_data)

    def __getitem__(self, idx):
        if not self.eval_mode:
            if idx < self.num_labeled_samples:
                raw_idx = idx // self.len_multiplier
                task_idx_unmapped = idx % self.len_multiplier
                task_idx = self.task_selection[task_idx_unmapped]
            else:
                raw_idx = self.modality_switch_index + (idx - self.num_labeled_samples)
                task_idx = 3

            emg_window, action_window, transition_index, coarse_action_window = (
                self.all_data[raw_idx]
            )

            if self.noise_amplitude > 0.0:
                amplitude = self.noise_amplitude
                noise = np.random.uniform(-amplitude, amplitude, size=emg_window.shape)
                emg_window = emg_window + noise
                emg_window = np.clip(emg_window, 0.0, 1.0)
            emg_window = emg_window.astype(np.float32)

            mask_type = (
                "poisson"
                if random.random() < self.sampling_probability_poisson
                else ("poisson" if task_idx == 3 else "end")
            )
            if mask_type == "poisson":
                mask_lower_bound, mask_upper_bound = (
                    self.poisson_mask_percentage_sampling_range[task_idx]
                )
            else:
                mask_lower_bound, mask_upper_bound = (
                    self.end_mask_percentage_sampling_range[task_idx]
                )

            if task_idx == 3:
                mask_percentage = random.uniform(
                    mask_lower_bound - 0.1, mask_upper_bound - 0.1
                )
            else:
                mask_percentage = random.uniform(mask_lower_bound, mask_upper_bound)

            selected_curriculum_stage = [0]
            if self.curriculum_stage == 1 and task_idx in [1, 3]:
                selected_curriculum_stage = random.choices([0, 1], self.stage_1_weights)
            if self.curriculum_stage == 2 and task_idx == 0:
                if (
                    transition_index >= self.transition_buffer
                    and transition_index <= self.window_size - self.transition_buffer
                ):
                    selected_curriculum_stage = [2]
                else:
                    selected_curriculum_stage = random.choices(
                        [0, 2], self.stage_2_weights
                    )

            action_mask_channel_selection = [0]
            use_bert_mask = False
            if selected_curriculum_stage == [0]:
                emg_mask_channel_selection = list(range(emg_window.shape[1]))
                use_bert_mask = True
            elif selected_curriculum_stage == [1]:
                mask_type = "end"
                mask_percentage = 1.0
                if self.embedding_method in ["linear_projection", "separate_channel"]:
                    emg_mask_channel_selection = random.sample(
                        list(range(emg_window.shape[1])), 2
                    )
                else:
                    raise Exception("unsupported embedding_method")
            elif selected_curriculum_stage == [2]:
                if (
                    transition_index > self.transition_buffer
                    and transition_index < self.window_size - self.transition_buffer
                ):
                    if random.choice([0, 1]) == 1:
                        mask_type = "targeted"
                    else:
                        mask_type = "end"
                        mask_percentage = 1.0
                else:
                    mask_type = "end"
                    mask_percentage = 1.0
            else:
                raise Exception("selected_curriculum_stage not recognized")

            emg_dummy_mask_positions = np.zeros_like(emg_window, dtype=bool)
            action_dummy_mask_positions = np.zeros_like(action_window, dtype=bool)
            emg_mask_token = self.mask_tokens_dict[self.embedding_method]["EMG_mask"]
            action_mask_token = self.mask_tokens_dict[self.embedding_method][
                "Action_mask"
            ]

            # FINE vs COARSE determination:
            if coarse_action_window is None:
                # Fine resolution
                if task_idx == 0:
                    masked_actions, mask_positions_actions = self.masking_a2a(
                        action_window,
                        mask_percentage,
                        mask_type,
                        action_mask_channel_selection,
                        self.mask_alignment,
                        self.lambda_poisson,
                        self.seeded_mask,
                        action_mask_token,
                        transition_index,
                        self.transition_buffer,
                        use_bert_mask,
                    )
                    masked_emg = emg_window
                    mask_positions_emg = emg_dummy_mask_positions
                elif task_idx == 1:
                    masked_emg, mask_positions_emg = self.masking_a2a(
                        emg_window,
                        mask_percentage,
                        mask_type,
                        emg_mask_channel_selection,
                        self.mask_alignment,
                        self.lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        self.transition_buffer,
                        use_bert_mask,
                    )
                    masked_actions = action_window
                    mask_positions_actions = action_dummy_mask_positions
                elif task_idx == 2:
                    masked_actions, mask_positions_actions = self.masking_a2a(
                        action_window,
                        mask_percentage,
                        mask_type,
                        action_mask_channel_selection,
                        self.mask_alignment,
                        self.lambda_poisson,
                        self.seeded_mask,
                        action_mask_token,
                        transition_index,
                        self.transition_buffer,
                        use_bert_mask,
                    )
                    masked_emg, mask_positions_emg = self.masking_a2a(
                        emg_window,
                        mask_percentage,
                        mask_type,
                        emg_mask_channel_selection,
                        self.mask_alignment,
                        self.lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        self.transition_buffer,
                        use_bert_mask,
                    )
                elif task_idx == 3:
                    masked_emg, mask_positions_emg = self.masking_a2a(
                        emg_window,
                        mask_percentage,
                        mask_type,
                        emg_mask_channel_selection,
                        self.mask_alignment,
                        self.lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        self.transition_buffer,
                        use_bert_mask,
                    )
                    masked_actions = np.full(action_window.shape, action_mask_token)
                    mask_positions_actions = action_dummy_mask_positions
                else:
                    raise Exception("task_idx not recognized")

                return (
                    emg_window,
                    action_window,
                    masked_emg,
                    masked_actions,
                    mask_positions_emg,
                    mask_positions_actions,
                    task_idx,
                    transition_index,
                )
            else:
                # Coarse resolution
                local_lambda_poisson = 2
                factor = self.window_size // self.inner_window_size
                local_transition_buffer = self.transition_buffer // factor
                if local_transition_buffer < 1:
                    local_transition_buffer = 1

                if task_idx == 0:
                    masked_coarse_actions, mask_positions_coarse_actions = (
                        self.masking_a2a(
                            coarse_action_window,
                            mask_percentage,
                            mask_type,
                            [0],
                            self.mask_alignment,
                            local_lambda_poisson,
                            self.seeded_mask,
                            action_mask_token,
                            transition_index,
                            local_transition_buffer,
                            use_bert_mask=True,
                        )
                    )
                    mask_positions_coarse_emg = np.zeros_like(
                        coarse_action_window, dtype=bool
                    )
                elif task_idx == 1:
                    dummy_coarse_emg = np.zeros_like(
                        coarse_action_window, dtype=np.int64
                    )
                    _, mask_positions_coarse_emg = self.masking_a2a(
                        dummy_coarse_emg,
                        mask_percentage,
                        mask_type,
                        [0],
                        self.mask_alignment,
                        local_lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        local_transition_buffer,
                        use_bert_mask=True,
                    )
                    masked_coarse_actions = coarse_action_window
                    mask_positions_coarse_actions = np.zeros_like(
                        coarse_action_window, dtype=bool
                    )
                elif task_idx == 2:
                    masked_coarse_actions, mask_positions_coarse_actions = (
                        self.masking_a2a(
                            coarse_action_window,
                            mask_percentage,
                            mask_type,
                            [0],
                            self.mask_alignment,
                            local_lambda_poisson,
                            self.seeded_mask,
                            action_mask_token,
                            transition_index,
                            local_transition_buffer,
                            use_bert_mask=True,
                        )
                    )
                    dummy_coarse_emg = np.zeros_like(
                        coarse_action_window, dtype=np.int64
                    )
                    _, mask_positions_coarse_emg = self.masking_a2a(
                        dummy_coarse_emg,
                        mask_percentage,
                        mask_type,
                        [0],
                        self.mask_alignment,
                        local_lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        local_transition_buffer,
                        use_bert_mask=True,
                    )
                elif task_idx == 3:
                    dummy_coarse_emg = np.zeros_like(
                        coarse_action_window, dtype=np.int64
                    )
                    _, mask_positions_coarse_emg = self.masking_a2a(
                        dummy_coarse_emg,
                        mask_percentage,
                        mask_type,
                        [0],
                        self.mask_alignment,
                        local_lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        local_transition_buffer,
                        use_bert_mask=True,
                    )
                    masked_coarse_actions = np.full_like(
                        coarse_action_window, fill_value=action_mask_token
                    )
                    mask_positions_coarse_actions = np.zeros_like(
                        coarse_action_window, dtype=bool
                    )
                else:
                    raise Exception("task_idx not recognized")

                return (
                    emg_window,
                    coarse_action_window,
                    masked_coarse_actions,
                    mask_positions_coarse_emg,
                    mask_positions_coarse_actions,
                    task_idx,
                    transition_index,
                )
        else:
            # Inference
            (emg_window, action_window, transition_index, coarse_action) = (
                self.all_data[idx]
            )
            untokenized_emg = self.untokenized_data[idx][0]

            if self.eval_task == "predict_action":
                task_idx = 0
            elif self.eval_task == "predict_emg":
                task_idx = 1
            elif self.eval_task == "predict_emg_ss":
                task_idx = 3
            else:
                task_idx = 0

            emg_dummy_mask_positions = np.zeros_like(emg_window, dtype=bool)
            action_dummy_mask_positions = np.zeros_like(action_window, dtype=bool)
            emg_mask_token = self.mask_tokens_dict[self.embedding_method]["EMG_mask"]
            action_mask_token = self.mask_tokens_dict[self.embedding_method][
                "Action_mask"
            ]
            use_bert_mask = False

            if self.window_size == self.inner_window_size:
                if self.eval_task == "predict_action":
                    masked_actions, mask_positions_actions = self.masking_a2a(
                        action_window,
                        1.0,
                        "end",
                        [0],
                        "non-aligned",
                        1,
                        self.seeded_mask,
                        action_mask_token,
                        transition_index,
                        self.transition_buffer,
                        use_bert_mask,
                    )
                    masked_emg = emg_window
                    mask_positions_emg = emg_dummy_mask_positions
                elif self.eval_task == "predict_emg":
                    if self.embedding_method == "linear_projection":
                        masked_emg, mask_positions_emg = self.masking_a2a(
                            emg_window,
                            self.eval_mask_percentage,
                            self.eval_mask_type,
                            range(emg_window.shape[1]),
                            "aligned",
                            1,
                            self.seeded_mask,
                            emg_mask_token,
                            transition_index,
                            self.transition_buffer,
                            use_bert_mask,
                        )
                        masked_actions = action_window
                        mask_positions_actions = action_dummy_mask_positions
                    else:
                        raise Exception(
                            "eval_task='predict_emg' but embedding_method not recognized"
                        )
                else:
                    raise Exception(
                        f"eval_task {self.eval_task} not recognized in inference."
                    )

                return (
                    emg_window,
                    action_window,
                    masked_emg,
                    masked_actions,
                    mask_positions_emg,
                    mask_positions_actions,
                    task_idx,
                    transition_index,
                    untokenized_emg,
                )
            else:
                coarse_len = self.window_size // self.inner_window_size
                local_lambda_poisson = 2
                local_transition_buffer = self.transition_buffer // coarse_len
                if local_transition_buffer < 1:
                    local_transition_buffer = 1

                dummy_coarse_emg = np.zeros((coarse_len,), dtype=np.int64)

                if self.eval_task == "predict_action":
                    masked_coarse_actions, mask_positions_coarse_actions = (
                        self.masking_a2a(
                            coarse_action,
                            1.0,
                            "end",
                            [0],
                            "non-aligned",
                            1,
                            self.seeded_mask,
                            action_mask_token,
                            transition_index,
                            local_transition_buffer,
                            use_bert_mask,
                        )
                    )
                    mask_positions_coarse_emg = np.zeros((coarse_len,), dtype=bool)
                elif self.eval_task == "predict_emg":
                    masked_coarse_emg, mask_positions_coarse_emg = self.masking_a2a(
                        dummy_coarse_emg,
                        self.eval_mask_percentage,
                        self.eval_mask_type,
                        [0],
                        "aligned",
                        local_lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        local_transition_buffer,
                        use_bert_mask,
                    )
                    masked_coarse_actions = coarse_action
                    mask_positions_coarse_actions = np.zeros((coarse_len,), dtype=bool)
                elif self.eval_task == "predict_emg_ss":
                    _, mask_positions_coarse_emg = self.masking_a2a(
                        dummy_coarse_emg,
                        self.eval_mask_percentage,
                        self.eval_mask_type,
                        [0],
                        "aligned",
                        local_lambda_poisson,
                        self.seeded_mask,
                        emg_mask_token,
                        transition_index,
                        local_transition_buffer,
                        use_bert_mask,
                    )
                    masked_coarse_actions = np.full_like(
                        coarse_action, fill_value=action_mask_token
                    )
                    mask_positions_coarse_actions = np.zeros((coarse_len,), dtype=bool)
                else:
                    raise Exception(
                        f"eval_task {self.eval_task} not recognized in coarse."
                    )

                return (
                    emg_window,
                    coarse_action,
                    masked_coarse_actions,
                    mask_positions_coarse_emg,
                    mask_positions_coarse_actions,
                    task_idx,
                    transition_index,
                    untokenized_emg,
                    action_window,
                )