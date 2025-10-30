import argparse
import os
import re
import json
import random as rdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from collections import Counter
from functools import partial
from torch.utils.data import DataLoader
from scipy.signal import medfilt
from dataset import (
    Any2Any_Dataset,
    EDTCN_Dataset,
    LSTM_Dataset,
    ANN_Dataset,
    TraHGR_Dataset
)
from nn_models import (
    Any2Any_Model,
    EDTCN_Model,
    LSTM_Model,
    ANN_Model,
    TraHGR_Model
)
from minlora import add_lora, merge_lora, LoRAParametrization
from torch.nn.utils import parametrize
from collections import defaultdict


###############################################################################
#                           HELPER FUNCTIONS
###############################################################################
def build_checkpoint_identifier(saved_checkpoint_pth: str) -> str:
    """
    Convert ".../run_folder/epoch_10.pth"  --->  "run_folder_epoch_10"
    """
    run_folder = os.path.basename(os.path.dirname(saved_checkpoint_pth))
    epoch_tag = os.path.splitext(os.path.basename(saved_checkpoint_pth))[0]
    return f"{run_folder}_{epoch_tag}"


def extract_buffer_windows(sequence, buffer_range):
    """
    Extracts buffer windows based on any class-to-class transition.
    """
    sequence = np.array(sequence)
    transition_indices = np.where(sequence[:-1] != sequence[1:])[0] + 1
    buffer_windows = []
    half_range = buffer_range // 2

    for i in range(len(transition_indices)):
        idx = transition_indices[i]
        start = max(0, idx - half_range)
        end = min(len(sequence), idx + half_range)

        if buffer_windows:
            prev_start, prev_end = buffer_windows[-1]
            if start <= prev_end:
                start = prev_end
        if start >= len(sequence) or start >= end:
            continue
        if i < len(transition_indices) - 1:
            next_transition = transition_indices[i + 1]
            if next_transition <= end:
                proposed_end = next_transition - 1
                if proposed_end <= start:
                    continue
                end = proposed_end

        if end > len(sequence):
            end = len(sequence)

        if start < end:
            buffer_windows.append((start, end))

    return buffer_windows


def extract_maintenance_windows(actions, buffer_windows):
    """
    Creates maintenance windows from buffer windows.
    Each maintenance window is from end of buffer i to start of buffer i+1.
    """
    if not buffer_windows:
        return [(0, len(actions))]

    maintenance_windows = []
    for i in range(len(buffer_windows) - 1):
        start = buffer_windows[i][1]  # End of buffer i
        end = buffer_windows[i + 1][0]  # Start of buffer i+1
        maintenance_windows.append((start, end))

    # Last maintenance window goes from end of last buffer to end of the entire sequence
    maintenance_windows.append((buffer_windows[-1][1], len(actions)))
    return maintenance_windows


def transition_metrics(
    prediction, ground_truth, buffer_windows, maintenance_windows, allow_relax
):
    """
    Checks correctness of events, each event = buffer window + maintenance window.
    Returns a transition accuracy in [0..1] and a list of reasons.
    """
    correct_transitions = 0
    total_transitions = 0
    transition_reasons = []

    # Special case: No buffer windows => no transitions found
    if not buffer_windows:
        # Usually means the ground_truth does not change class at all.
        unique_gt = np.unique(ground_truth)
        if len(unique_gt) == 1:
            single_label = unique_gt[0]
            # Check if the entire prediction is the same label
            if np.all(prediction == single_label):
                return 1.0, ["Successful"]
            else:
                return 0.0, [
                    "No transitions but prediction does not match single GT label."
                ]
        else:
            # If ground_truth has multiple labels but still no buffer windows,
            # that is contradictory data.
            raise Exception(
                "No buffer windows, but ground_truth has multiple labels (data error?)"
            )

    # Iterate over each event i
    for i in range(len(buffer_windows)):
        total_transitions += 1

        # --------------------
        # Extract buffer + maintenance
        # --------------------
        buffer_start, buffer_end = buffer_windows[i]
        maintenance_start, maintenance_end = maintenance_windows[i]

        buffer_gt = ground_truth[buffer_start:buffer_end]
        buffer_pred = prediction[buffer_start:buffer_end]
        maintenance_gt = ground_truth[maintenance_start:maintenance_end]
        maintenance_pred = prediction[maintenance_start:maintenance_end]

        # ----------------------------------------------------
        # 1) Identify which transition (A -> B) in ground_truth
        #    We look for the first ground-truth "A->B" in buffer_gt (A != B).
        # ----------------------------------------------------
        valid_gt_transition_indices = np.where(buffer_gt[:-1] != buffer_gt[1:])[0]
        if len(valid_gt_transition_indices) == 0:
            raise Exception("Error: No valid transition in ground-truth buffer.")

        idx = valid_gt_transition_indices[0]
        A = buffer_gt[idx]
        B = buffer_gt[idx + 1]

        # ----------------------------------------------------
        # 2) Check predicted buffer
        #    Must contain at least one A->B transition.
        #    Allowed classes depend on allow_relax.
        # ----------------------------------------------------
        if allow_relax == 1:
            allowed_classes_buffer = {A, B, 0}
        else:
            allowed_classes_buffer = {A, B}

        unique_buffer_pred = set(np.unique(buffer_pred))
        if not unique_buffer_pred.issubset(allowed_classes_buffer):
            transition_reasons.append(
                "Incorrect: Buffer prediction has classes outside allowed set."
            )
            continue

        valid_pred_transition_indices = np.where(
            (buffer_pred[:-1] == A) & (buffer_pred[1:] == B)
        )[0]
        if len(valid_pred_transition_indices) == 0:
            transition_reasons.append(
                "Incorrect: No predicted A->B transition in buffer."
            )
            continue

        # 0-length maintenance windows are considered correct
        if len(maintenance_gt) == 0:
            correct_transitions += 1
            transition_reasons.append("Successful")
            continue

        # ----------------------------------------------------
        # 3) Maintenance window must be purely the new class B
        #    (or B + 0 if allow_relax == 1)
        # ----------------------------------------------------
        unique_maintenance_gt = np.unique(maintenance_gt)
        if len(unique_maintenance_gt) != 1:
            raise Exception("Error: Maintenance window GT not a single intent.")
        gt_maintenance_class = unique_maintenance_gt[0]
        if gt_maintenance_class != B:
            raise Exception("Error: maintenance window GT != B.")

        if len(maintenance_pred) > 0:
            if allow_relax == 1:
                allowed_classes_maintenance = {B, 0}
                unique_maintenance_pred = set(np.unique(maintenance_pred))
                if not unique_maintenance_pred.issubset(allowed_classes_maintenance):
                    transition_reasons.append(
                        "Incorrect: Maintenance pred has classes outside {B, 0}."
                    )
                    continue
                if B not in unique_maintenance_pred:
                    transition_reasons.append(
                        "Incorrect: Maintenance window has only 0, no B predicted."
                    )
                    continue
            else:
                if not np.all(maintenance_pred == B):
                    transition_reasons.append(
                        "Incorrect: Maintenance window not strictly class B."
                    )
                    continue

        correct_transitions += 1
        transition_reasons.append("Successful")

    transition_accuracy = (
        (correct_transitions / total_transitions) if total_transitions > 0 else 0.0
    )
    return transition_accuracy, transition_reasons


def build_gt_sequence(
    gt_matrix,
    stride,
    window_size,
    total_timesteps,
    lookahead,
    samples_between_prediction,
):
    """
    Builds a 1D ground-truth array from the 2D GT matrix.
    For t >= total_timesteps - lookahead, we set GT = -1.
    """
    num_windows, _ = gt_matrix.shape
    gt_aggregated = np.full(total_timesteps, -1, dtype=int)

    # Reconstruct
    for w_idx in range(num_windows):
        start_i = w_idx * stride
        end_i = start_i + window_size
        if end_i > total_timesteps:
            end_i = total_timesteps
        for t in range(start_i, end_i):
            if gt_aggregated[t] == -1 and gt_matrix[w_idx, t] != -1:
                gt_aggregated[t] = gt_matrix[w_idx, t]

    # Finally, zero-out the last lookahead region because there's not enough future data
    cutoff = max(0, total_timesteps - lookahead)
    gt_aggregated[cutoff:] = -1
    return gt_aggregated


def realtime_online_aggregation(
    pred_matrix,  # shape: [num_windows, total_timesteps],  integer predictions (argmax) per window & timestep
    logits_matrix,  # shape: [num_windows, total_timesteps, num_classes], raw logits or probabilities
    window_size,
    stride,
    lookahead,
    samples_between_prediction,
    maj_vote_range,
    likelihood_format,
    weight_max_factor,
):
    """
    Produces a final 1D array of shape (total_timesteps,) with real-time style aggregation.

    Parameters
    ----------
    pred_matrix : np.ndarray of shape (num_windows, total_timesteps)
        The argmax-based predictions (class IDs). Each row i corresponds
        to the i-th window, and columns are timesteps in the global index.
        Entries are -1 if that window does not cover that timestep.
    logits_matrix : np.ndarray of shape (num_windows, total_timesteps, num_classes) or None
        The raw logits (or probabilities) per window & timestep.
        If likelihood_format=='argmax', logits_matrix can be None or ignored.
        If likelihood_format in {'logits', 'probs'}, we need this non-None.
        The shape uses -1 for invalid coverage as needed.
    window_size : int
        Number of timesteps in each window.
    stride : int
        Offset in timesteps between consecutive windows.
    lookahead : int
        Number of future timesteps (beyond the current t) to consider for the 'future' case.
    samples_between_prediction : int
        We produce a new prediction every 'samples_between_prediction' timesteps.
        In between, we hold the old prediction.
    maj_vote_range : str, one of {'single', 'future'}
        - 'single': Aggregate only the predictions (or logits) for the *single* global timestep t.
        - 'future': Aggregate the predictions (or logits) for *all* timesteps in [t.. t+lookahead].
    likelihood_format : str, one of {'argmax', 'logits'}
        - 'argmax': We do a majority vote over the argmax from each window/timestep
        - 'logits': We do an (optionally weighted) average of the logits, then argmax.
    weight_max_factor : float
        How large the weighting becomes for the *latest* window among those considered.
        E.g. if we have N=5 windows in the future set, the earliest has weight=1.0,
        the last has weight=weight_max_factor, and we linearly space between them.

    Returns
    -------
    aggregated : np.ndarray of shape (total_timesteps,)
        The final aggregated labels for each global timestep.
        For t >= total_timesteps - lookahead, we fill with -1 because
        we cannot produce a valid future-based prediction that looks
        ahead. We also only update the label once every
        'samples_between_prediction' timesteps.
    """
    num_windows, total_timesteps = pred_matrix.shape
    aggregated = np.full(total_timesteps, -1, dtype=int)

    # ------------------------------------------------------------
    # Identify the coverage of each window i: it covers [start_i, end_i)
    # where start_i = i * stride, end_i = start_i + window_size (clipped).
    # We'll store these for quick lookup.
    # ------------------------------------------------------------
    window_ranges = []
    for i in range(num_windows):
        start_i = i * stride
        end_i = start_i + window_size
        if end_i > total_timesteps:
            end_i = total_timesteps
        window_ranges.append((start_i, end_i))

    # Main loop:
    # For t in [0 .. total_timesteps - lookahead), step by samples_between_prediction,
    #   compute a single label "label_t"
    # Then fill aggregator[t .. t+samples_between_prediction-1] with label_t
    # (holding it until the next prediction).
    # ------------------------------------------------------------
    # We will not produce predictions for the last 'lookahead' frames, i.e. for
    # t >= (total_timesteps - lookahead). aggregator remains -1 there.
    # ------------------------------------------------------------
    last_label = -1
    max_t_for_predictions = max(
        0, total_timesteps - lookahead
    )  # if total_timesteps < lookahead, becomes 0

    t = 0
    while t < max_t_for_predictions:
        # -------------------------------------------------------------------
        # Only consider windows whose *last covered index* (we-1) is
        # within [t .. t+lookahead].
        # Then, for each valid window, we gather either:
        #   - 'single':  the prediction/logits at time t,
        #   - 'future':  the predictions/logits for [t.. min(we, t+lookahead)].
        # -------------------------------------------------------------------
        lookahead_end = t + lookahead
        valid_windows_idx = []
        for w_idx, (ws, we) in enumerate(window_ranges):
            last_covered = we - 1  # the final index covered by this window
            # we use this window if it ends in [t .. lookahead_end]
            if last_covered >= t and last_covered <= lookahead_end:
                valid_windows_idx.append(w_idx)

        # minimal change to allow using the first window for t < window_size
        if len(valid_windows_idx) == 0 and t < window_size:
            valid_windows_idx.append(0)

        # Sort them by window start time (for weighting)
        valid_windows_idx.sort(key=lambda x: window_ranges[x][0])

        # Build a list of tuples: (w_idx, ov_start, ov_end, ov_len)
        valid_windows = []
        if maj_vote_range == "single":
            for w_idx in valid_windows_idx:
                ov_start = t
                ov_end = t + 1  # gather exactly [t..t+1)
                ov_len = 1
                valid_windows.append((w_idx, ov_start, ov_end, ov_len))
        else:  # maj_vote_range == 'future'
            for w_idx in valid_windows_idx:
                ws, we = window_ranges[w_idx]
                ov_start = t
                ov_end = min(we, lookahead_end)
                if ov_end > ov_start:
                    ov_len = ov_end - ov_start
                    valid_windows.append((w_idx, ov_start, ov_end, ov_len))

        # If no windows matched, default to last_label or -1
        if len(valid_windows) == 0:
            label_t = last_label
            if label_t == -1:
                # if we haven't assigned anything yet, let's skip or keep it -1
                label_t = -1
        else:
            # ----------------------------------------------------
            # 2) Depending on likelihood_format, do majority vote or
            #    (weighted) average of logits
            # ----------------------------------------------------
            if likelihood_format == "argmax":
                # (a) Gather all predicted labels from each window in the region.
                all_labels = []
                for w_idx, ov_start, ov_end, ov_len in valid_windows:
                    if maj_vote_range == "single":
                        # just the single time 't'
                        pred_label = pred_matrix[w_idx, t]
                        if pred_label != -1:
                            all_labels.append(pred_label)
                    else:
                        # 'future' mode: gather pred_matrix[w_idx, s] for s in [ov_start..ov_end)
                        for s in range(ov_start, ov_end):
                            pred_label = pred_matrix[w_idx, s]
                            if pred_label != -1:
                                all_labels.append(pred_label)

                if len(all_labels) == 0:
                    label_t = -1  # nothing valid
                else:
                    # majority vote
                    counts = np.bincount(all_labels)
                    label_t = np.argmax(counts)

            elif likelihood_format == "logits":
                # We combine raw logits. Potentially apply weighting from 1 to weight_max_factor.
                # Weighted approach: earliest window has weight=1, last=weight_max_factor, linear in between.

                num_windows_in_region = len(valid_windows)
                if num_windows_in_region == 1:
                    # Only one window => easy
                    w_idx, ov_start, ov_end, ov_len = valid_windows[0]
                    w = 1.0

                    # We'll sum logits across [ov_start.. ov_end), multiply by w
                    sum_logits = np.zeros(logits_matrix.shape[-1], dtype=np.float64)
                    total_time = 0.0
                    for s in range(ov_start, ov_end):
                        if s < total_timesteps and logits_matrix[w_idx, s, 0] != -1:
                            sum_logits += logits_matrix[w_idx, s]
                            total_time += 1.0
                    if total_time == 0:
                        label_t = -1
                    else:
                        avg_logits = (sum_logits * w) / (total_time * w)
                        label_t = np.argmax(avg_logits)
                else:
                    # Multiple windows
                    sum_logits_all = np.zeros(logits_matrix.shape[-1], dtype=np.float64)
                    total_weighted_time = 0.0

                    # Precompute window weights
                    # For j in [0..N-1], weight_j = 1 + j*(weight_max_factor-1)/(N-1)
                    weights = []
                    for j in range(num_windows_in_region):
                        if num_windows_in_region == 1:
                            w = 1.0
                        else:
                            w = 1.0 + j * (weight_max_factor - 1.0) / (
                                num_windows_in_region - 1
                            )
                        weights.append(w)

                    for j, (w_idx, ov_start, ov_end, ov_len) in enumerate(
                        valid_windows
                    ):
                        w = weights[j]
                        # Summation of logits over overlap region
                        sum_logits_win = np.zeros(
                            logits_matrix.shape[-1], dtype=np.float64
                        )
                        time_count = 0.0

                        for s in range(ov_start, ov_end):
                            # Check validity
                            if s < total_timesteps and logits_matrix[w_idx, s, 0] != -1:
                                sum_logits_win += logits_matrix[w_idx, s]
                                time_count += 1.0

                        if time_count > 0:
                            sum_logits_all += w * sum_logits_win
                            total_weighted_time += w * time_count

                    if total_weighted_time == 0:
                        label_t = -1
                    else:
                        avg_logits = sum_logits_all / total_weighted_time
                        label_t = np.argmax(avg_logits)
            else:
                raise ValueError(f"Unsupported likelihood_format = {likelihood_format}")

        # The newly computed label
        new_label = label_t
        if new_label == -1 and last_label != -1:
            # Optionally hold old label if new_label is -1
            new_label = last_label

        # --------------------------------------------------------
        # 3) Fill aggregator[t : t+samples_between_prediction] with new_label
        #    (unless it goes beyond the valid range).
        # --------------------------------------------------------
        fill_end = t + samples_between_prediction
        if fill_end > max_t_for_predictions:
            fill_end = max_t_for_predictions
        aggregated[t:fill_end] = new_label

        last_label = new_label
        t += samples_between_prediction

    # For t >= max_t_for_predictions, we leave aggregated as -1.
    return aggregated


def repeat_chunks(
    tensor,
    original_length,
    inner_window_size,
    inner_stride,
):
    """
    Upsamples a reduced output of shape (B, T_red[, D]) to (B, original_length[, D]),
    following a specific overlapping window logic:

      - The 1st label covers timesteps [0 : inner_window_size).
      - The 2nd label covers timesteps [inner_window_size : inner_window_size + inner_stride).
      - The 3rd label covers timesteps [inner_window_size + inner_stride : inner_window_size + 2*inner_stride),
        and so on, until all T_red labels have been assigned.

    It's assumed that:
        inner_window_size + (T_red - 1)*inner_stride == original_length

    Args:
        tensor (torch.Tensor):
            2D shape = (B, T_red)
            OR
            3D shape = (B, T_red, D)
        original_length (int):
            Desired final temporal length of the upsampled sequence.
        inner_window_size (int):
            The size of the first window (in original timesteps).
        inner_stride (int):
            The stride used for subsequent windows (in original timesteps).

    Returns:
        upsampled (torch.Tensor):
            shape = (B, original_length) if input was 2D
            shape = (B, original_length, D) if input was 3D
    """
    # Determine input dimensions
    dim = tensor.dim()
    if dim == 2:
        # tensor shape = (B, T_red)
        B, T_red = tensor.shape
        # Prepare output placeholder
        upsampled = tensor.new_zeros((B, original_length))
    elif dim == 3:
        # tensor shape = (B, T_red, D)
        B, T_red, D = tensor.shape
        upsampled = tensor.new_zeros((B, original_length, D))
    else:
        raise ValueError("repeat_chunks expects a 2D or 3D tensor")

    # Fill the output according to the described logic
    for i in range(T_red):
        if i == 0:
            # First label -> covers [0 : inner_window_size)
            start_idx = 0
            end_idx = inner_window_size
        else:
            # i-th label -> covers [inner_window_size + (i-1)*inner_stride : inner_window_size + i*inner_stride)
            start_idx = inner_window_size + (i) * inner_stride
            end_idx = start_idx + inner_stride

        # In case there's any rounding or mismatch, clamp end_idx
        if end_idx > original_length:
            end_idx = original_length

        if end_idx <= start_idx:
            # Safety check: no more space to fill
            break

        # Fill upsampled with the i-th label
        if dim == 2:
            # 2D: shape = (B, T_red)
            upsampled[:, start_idx:end_idx] = tensor[:, i].unsqueeze(-1)
        else:
            # 3D: shape = (B, T_red, D)
            upsampled[:, start_idx:end_idx, :] = tensor[:, i, :].unsqueeze(1)

    return upsampled


def evaluate_predictions(
    predicted_actions,
    ground_truth_actions,
    buffer_range,
    saved_checkpoint_pth,
    allow_relax,
):
    """
    Evaluates using transition_metrics + overall accuracy. Returns a results dictionary.
    """
    buffer_windows = extract_buffer_windows(ground_truth_actions, buffer_range)
    maintenance_windows = extract_maintenance_windows(
        ground_truth_actions, buffer_windows
    )
    transition_acc, transition_reasons = transition_metrics(
        predicted_actions,
        ground_truth_actions,
        buffer_windows,
        maintenance_windows,
        allow_relax,
    )
    overall_accuracy = np.mean(predicted_actions == ground_truth_actions)
    event_counts = len(buffer_windows)

    results = {
        "transition_window_accuracy": transition_acc,
        "overall_accuracy": overall_accuracy,
        "transition_reasons": transition_reasons,
        "event_counts": event_counts,
    }
    return results


def plot_event_classification(
    path,
    pred_sequence,
    gt_sequence,
    downsample,
    args_dict,
    saved_checkpoint_pth,
    results,
    lookahead,
    verbose,
):
    """
    Loads EMG data from CSV or NPY, applies preprocessing, and plot EMG (top), GT (middle), and Pred (bottom)
    Saves to:
      output/<checkpoint_folder_name>/plots/event_<parent_folder_name>_<base_filename>.png
    """

    # if not verbose, skip plotting
    if verbose == 0:
        return

    # Construct folder names
    checkpoint_identifier = build_checkpoint_identifier(saved_checkpoint_pth)
    folder_name = f"{checkpoint_identifier}_LA{lookahead}"
    summary_dir = os.path.join("output", folder_name)
    base_dir = os.path.join(summary_dir, "plots")
    os.makedirs(base_dir, exist_ok=True)

    base_filename = os.path.splitext(os.path.basename(path))[0]
    save_filename = f"{folder_name}_{base_filename}.png"
    save_path = os.path.join(base_dir, save_filename)

    # Load csv or npy
    # npy column format: 'gt', followed by 'emg0' through 'emg7'
    # Median filter + rectification + scale
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
        try:
            df_emg = df[
                ["emg_0", "emg_1", "emg_2", "emg_3", "emg_4", "emg_5", "emg_6", "emg_7"]
            ]
        except KeyError:
            df_emg = df[
                ["emg0", "emg1", "emg2", "emg3", "emg4", "emg5", "emg6", "emg7"]
            ]
        df_emg = df_emg.apply(
            lambda x: medfilt(x, kernel_size=args_dict["median_filter_size"])
        )
        df_emg = np.abs(df_emg / 128).astype(np.float32)
        df_npy = df_emg.to_numpy()
    elif path.lower().endswith(".npy"):
        loaded = np.load(path).astype(np.float32)
        data_array = loaded[:, 1:9]
        filtered_array = np.empty_like(data_array)
        for c in range(data_array.shape[1]):
            filtered_array[:, c] = medfilt(
                data_array[:, c], kernel_size=args_dict["median_filter_size"]
            )
        filtered_array = np.abs(filtered_array / 128.0).astype(np.float32)
        df_npy = filtered_array
    else:
        raise ValueError(
            "File extension not recognized. Must be .csv or .npy for this function."
        )

    # Crop EMG to match the final sequence length, so the plot lines up with GT/pred; Should not be necessary
    df_npy = df_npy[: len(pred_sequence)]

    # Plot construction
    fig, axs = plt.subplots(
        3, 1, figsize=(20, 24), gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    # 1) Plot EMG signals
    for ch in range(df_npy.shape[1]):
        axs[0].plot(df_npy[:, ch], label=f"Channel {ch}", alpha=0.7)
    axs[0].set_title(
        f"{base_filename} - Event Acc={results['transition_window_accuracy']:.2f}; "
        f"Num Events={results['event_counts']}",
        fontsize=18,
    )
    axs[0].set_ylabel("EMG Value", fontsize=18)
    axs[0].legend(loc="upper right", fontsize=18)
    axs[0].tick_params(axis="x", labelsize=18)
    axs[0].tick_params(axis="y", labelsize=18)

    # ----------------------------
    # 2) Plot GT (middle)
    # ----------------------------
    axs[1].plot(gt_sequence, label="GT Sequence", color="green", alpha=0.7)
    axs[1].set_ylabel("GT Value", fontsize=18)
    axs[1].legend(loc="upper right", fontsize=18)
    axs[1].tick_params(axis="x", labelsize=18)
    axs[1].tick_params(axis="y", labelsize=18)
    # Use custom y-tick labels for 0,1,2 -> "relax","open","close"
    axs[1].set_yticks([0, 1, 2])
    axs[1].set_yticklabels(["relax", "open", "close"], fontsize=18)

    # ----------------------------
    # 3) Plot Predictions (bottom)
    # ----------------------------
    axs[2].plot(pred_sequence, label="Pred Sequence", color="orange", alpha=0.7)
    axs[2].set_xlabel("Timestep", fontsize=18)
    axs[2].set_ylabel("Pred Value", fontsize=18)
    axs[2].legend(loc="upper right", fontsize=18)
    axs[2].tick_params(axis="x", labelsize=18)
    axs[2].tick_params(axis="y", labelsize=18)
    # Same custom y-tick labels for 0,1,2 -> "relax","open","close"
    axs[2].set_yticks([0, 1, 2])
    axs[2].set_yticklabels(["relax", "open", "close"], fontsize=18)

    # Ensure margins are sufficient so text does not get cut off
    plt.tight_layout(pad=2.0)

    # Save figure
    plt.savefig(save_path)
    plt.close()


def initialize_dataset(
    model_choice,
    args_dict,
    csv_path,
    stride,
    eval_task,
    transition_samples_only,
    mask_percentage,
    mask_type,
):
    if model_choice == "any2any":
        dataset = Any2Any_Dataset(
            labeled_csv_paths=[csv_path],
            unlabeled_csv_paths=[],  # Inference => no unlabeled
            median_filter_size=args_dict["median_filter_size"],
            window_size=args_dict["window_size"],
            offset=stride,
            embedding_method=args_dict["embedding_method"],
            lambda_poisson=1,  # Not really used in inference
            seeded_mask=True,
            sampling_probability_poisson=1.0,  # Also not used in inference
            poisson_mask_percentage_sampling_range=[(0.0, 0.0)] * 5,  # placeholders
            end_mask_percentage_sampling_range=[(0.0, 0.0)] * 5,  # placeholders
            task_selection=[0],  # Specifying task 0 only for eval_task action
            stage_1_weights=[1, 1],
            stage_2_weights=[1, 1],
            mask_alignment="non-aligned",
            transition_buffer=args_dict["transition_buffer"],
            mask_tokens_dict=args_dict["mask_tokens_dict"],
            with_training_curriculum=False,
            num_classes=args_dict["num_classes"],
            medfilt_order=args_dict["medfilt_order"],
            noise=0.0,
            hand_choice=args_dict["hand_choice"],
            inner_window_size=args_dict["inner_window_size"],
            use_mav_for_emg=args_dict["use_mav_for_emg"],
            # Now the inference-only arguments
            eval_mode=True,
            eval_task=eval_task,
            transition_samples_only=transition_samples_only,
            mask_percentage=mask_percentage,
            mask_type=mask_type,
        )
    elif model_choice == "ed_tcn":
        dataset = EDTCN_Dataset(
            window_size=args_dict["window_size"],
            offset=stride,
            file_paths=[csv_path],
            inner_window_size=150,
            inner_stride=25,
        )
    elif model_choice == "lstm":
        dataset = LSTM_Dataset(
            window_size=args_dict["window_size"],
            offset=stride,
            csv_paths=[csv_path],
            num_classes=args_dict["num_classes"],
            precomputed_mean=args_dict["precomputed_mean"],
            precomputed_std=args_dict["precomputed_std"],
        )
    elif model_choice == "ann":
        dataset = ANN_Dataset(
            window_size=args_dict["window_size"],
            offset=stride,
            file_paths=[csv_path],
            num_classes=args_dict["num_classes"],
            use_precomputed_stats=True,
            precomputed_mean=args_dict["precomputed_mean"],
            precomputed_std=args_dict["precomputed_std"],
        )
    elif model_choice == "trahgr":
        dataset = TraHGR_Dataset(
            window_size=args_dict["window_size"],
            offset=stride,
            file_paths=[csv_path],
            num_classes=args_dict["num_classes"],
            butter_cutoff_hz=args_dict.get("trahgr_butter_cutoff_hz", 90.0),
        )
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    return dataset


def initialize_model(args_dict, checkpoint, model_choice, device):
    if model_choice == "any2any":
        model = Any2Any_Model(
            args_dict["embedding_dim"],
            args_dict["nhead"],
            args_dict["dropout"],
            args_dict["activation"],
            args_dict["num_layers"],
            args_dict["window_size"],
            args_dict["embedding_method"],
            args_dict["mask_alignment"],
            args_dict["share_pe"],
            args_dict["tie_weight"],
            args_dict["use_decoder"],
            args_dict["use_input_layernorm"],
            args_dict["num_classes"],
            args_dict["output_reduction_method"],
            args_dict["chunk_size"],
            600,
            0,
            1,
        )
        if args_dict["use_lora"] == 1:
            lora_config = {
                nn.Linear: {
                    "weight": partial(
                        LoRAParametrization.from_linear,
                        rank=args_dict["lora_rank"],
                        lora_alpha=args_dict["lora_alpha"],
                        lora_dropout_p=args_dict["lora_dropout_p"],
                    ),
                },
            }
            add_lora(model, lora_config)
    elif model_choice == "ed_tcn":
        model = EDTCN_Model(
            num_channels=8,
            num_classes=args_dict["num_classes"],
            enc_filters=(128, 288),
            kernel_size=9,
        )
    elif model_choice == "lstm":
        model = LSTM_Model(
            input_size=8,
            fc_size=400,
            hidden_size=256,
            num_classes=args_dict["num_classes"],
        )
    elif model_choice == "ann":
        model = ANN_Model(num_classes=args_dict["num_classes"])
    elif model_choice == "trahgr":
        # Defaults = TraHGR-Huge from the paper (L=1, D=144, MLP=720, h=8)
        model = TraHGR_Model(
            num_classes=args_dict["num_classes"],
            embed_dim=args_dict["embedding_dim"],
            num_heads=args_dict["nhead"],
            num_layers=args_dict["num_layers"],
            dropout=args_dict["dropout"],
            window_len=args_dict["window_size"],
            num_sensors=8,
            num_filter_orders=3

        )
        print("hi")

        
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    model.load_state_dict(checkpoint["model_info"]["model_state_dict"], strict=False)
    if "use_lora" in args_dict and args_dict["use_lora"] == 1:
        model.load_state_dict(checkpoint["lora_state_dict"], strict=False)
        # minLoRA monkey-patch
        # Since PyTorch 2.1 the iterator returned by dict.keys() becomes invalid as soon
        # as the dict size changes, so Python raises "RuntimeError: dictionary changed size during iteration"
        safe_merge_lora(model)

    model.to(device)
    model.eval()

    return model


def _safe_merge(layer):
    if hasattr(layer, "parametrizations"):
        for k in list(layer.parametrizations.keys()):
            parametrize.remove_parametrizations(layer, k, leave_parametrized=True)


def safe_merge_lora(model):
    model.apply(_safe_merge)


def process_and_evaluate(
    model,
    dataset_eval,
    device,
    eval_batch_size,
    eval_task,
    csv_path,
    downsample,
    args_dict,
    buffer_range,
    saved_checkpoint_pth,
    allow_relax,
    lookahead,
    weight_max_factor,
    likelihood_format,
    samples_between_prediction,
    maj_vote_range,
    stride,
    epn_eval,
    recog_threshold,
    verbose,
    model_choice,
):
    """
    Runs inference for a single file, returns:
      (results_dict, pred_aggregated, gt_aggregated)
    """

    if model_choice == "ann":
        # We assume dataset_eval.__getitem__ returns (features_48d, label_int, raw_gt_seq)
        #   features_48d: shape [48], the ANN input features
        #   label_int:    int label for that window
        #   raw_gt_seq:   shape [window_size], the ground-truth labels for every timestep in that window

        # We create a DataLoader
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )

        model.eval()
        window_size = args_dict["window_size"]
        num_windows = len(dataset_eval)

        # We define a final timeline for the *real* signal of length:
        #   final_length = (num_windows - 1)*stride + window_size
        # Because the 1st window ends at index (window_size - 1),
        #   the 2nd window ends at (stride + window_size - 1),
        #   etc.
        final_length = (num_windows - 1) * stride + window_size

        # Initialize with -1 to indicate "no prediction yet"
        pred_aggregated = np.full(final_length, -1, dtype=int)
        gt_aggregated = np.full(final_length, -1, dtype=int)

        last_filled_t = -1  # last time index that we assigned a prediction
        last_pred = -1  # the most recent predicted label

        with torch.no_grad():
            # We'll read each sample in sequential order.
            for global_i, batch_data in enumerate(dataloader_eval):
                # batch_data is (X, y, raw_gt_seq), but since we have a batch, let's parse them
                X_batch, y_batch, raw_gt_batch = batch_data
                #   X_batch:  shape [B, 48]
                #   y_batch:  shape [B]
                #   raw_gt_batch: shape [B, window_size]

                # Forward pass: shape [B, num_classes]
                logits = model(X_batch.to(device))
                preds = torch.argmax(logits, dim=-1).cpu().numpy()  # shape [B]

                # We'll loop through each item in the batch
                for b in range(X_batch.shape[0]):
                    # The "i-th window" in the dataset is: i = global_i*eval_batch_size + b
                    i_window = global_i * eval_batch_size + b
                    if i_window >= num_windows:
                        break  # in case the last batch is partially filled

                    # The window's "end" is (i_window*stride + (window_size - 1))
                    end_t = i_window * stride + (window_size - 1)
                    if end_t >= final_length:
                        continue

                    current_pred = preds[b]  # single label for this window
                    # For ground truth, we have raw_gt_batch[b], shape [window_size].
                    # We want to fill the portion that belongs to new timesteps we haven't assigned yet.

                    # 1) Fill predictions from last_filled_t+1 up to end_t with the "most recent known pred"
                    for fill_t in range(last_filled_t + 1, end_t + 1):
                        if fill_t < 0 or fill_t >= final_length:
                            continue
                        # If we have no "previous" prediction, use the current window pred
                        pred_aggregated[fill_t] = (
                            last_pred if last_pred != -1 else current_pred
                        )

                    # Now update "last_pred" and "last_filled_t"
                    last_pred = current_pred
                    last_filled_t = end_t

                    # 2) Fill ground truth for [last_filled_t+1 .. end_t] using the last `stride` part of raw_gt_seq
                    #    "the new non-overlapping timesteps should be used for ground truth construction"
                    # We'll interpret the user request: the "last stride" portion is raw_gt_seq[b, window_size - stride : ].
                    # We'll place it in [end_t - stride + 1 .. end_t], or from the old last_filled_t..end_t
                    # Example approach:

                    # The slice in raw_gt_seq
                    if stride > window_size:
                        # edge case: if stride > window_size, the "last stride portion" is bigger than the window itself
                        # pick the entire raw_gt_seq
                        relevant_gt = raw_gt_batch[b].numpy()  # shape [window_size]
                    else:
                        start_idx = max(0, window_size - stride)
                        relevant_gt = raw_gt_batch[
                            b, start_idx:
                        ].numpy()  # shape [stride] typically

                    stride_len = len(relevant_gt)
                    # The segment in global GT we want to fill
                    gt_fill_start = end_t - (stride_len - 1)
                    if gt_fill_start < 0:
                        gt_fill_start = 0
                    # Fill it
                    for k in range(stride_len):
                        fill_t = gt_fill_start + k
                        if fill_t >= final_length:
                            break
                        gt_aggregated[fill_t] = relevant_gt[k]

        # After finishing all windows, we may have leftover timesteps from last_filled_t+1 .. final_length-1
        # that remain -1. The user wants us to keep carrying forward the last prediction:
        if last_pred != -1:
            for fill_t in range(last_filled_t + 1, final_length):
                pred_aggregated[fill_t] = last_pred

        # trim empty filler at the beginning
        gt_aggregated = gt_aggregated[window_size:]
        pred_aggregated = pred_aggregated[window_size:]

    elif model_choice == "trahgr":
        # TraHGR: treat as single-label window classifier (use fused head)
        # dataset_eval.__getitem__ returns: (temporal_patches [S, W*C], featural_patches [Nf, S*S*C], y, raw_gt_seq)
        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )

        model.eval()
        window_size = args_dict["window_size"]
        num_windows = len(dataset_eval)

        final_length = (num_windows - 1) * stride + window_size
        pred_aggregated = np.full(final_length, -1, dtype=int)
        gt_aggregated = np.full(final_length, -1, dtype=int)

        last_filled_t = -1
        last_pred = -1

        with torch.no_grad():
            for global_i, batch_data in enumerate(dataloader_eval):
                temporal_patches, featural_patches, y_batch, raw_gt_batch = batch_data
                # Forward: get fused logits (y_fused)
                y_fused, y_tnet, y_fnet = model(
                    temporal_patches.to(device).float(),
                    featural_patches.to(device).float()
                )
                preds = torch.argmax(y_fused, dim=-1).cpu().numpy()  # [B]

                for b in range(temporal_patches.size(0)):
                    i_window = global_i * eval_batch_size + b
                    if i_window >= num_windows:
                        break

                    end_t = i_window * stride + (window_size - 1)
                    if end_t >= final_length:
                        continue

                    current_pred = int(preds[b])

                    # Fill predictions from the last assigned index up to end_t
                    for fill_t in range(last_filled_t + 1, end_t + 1):
                        if 0 <= fill_t < final_length:
                            pred_aggregated[fill_t] = last_pred if last_pred != -1 else current_pred

                    # Update rolling state
                    last_pred = current_pred
                    last_filled_t = end_t

                    # Fill GT for the newly added part using the last `stride` timesteps of the window
                    if stride > window_size:
                        relevant_gt = raw_gt_batch[b].numpy()
                    else:
                        start_idx = max(0, window_size - stride)
                        relevant_gt = raw_gt_batch[b, start_idx:].numpy()

                    stride_len = len(relevant_gt)
                    gt_fill_start = end_t - (stride_len - 1)
                    gt_fill_start = max(0, gt_fill_start)
                    for k in range(stride_len):
                        fill_t = gt_fill_start + k
                        if fill_t >= final_length:
                            break
                        gt_aggregated[fill_t] = int(relevant_gt[k])

        # Carry forward last prediction to the end if needed
        if last_pred != -1:
            for fill_t in range(last_filled_t + 1, final_length):
                pred_aggregated[fill_t] = last_pred

        # Trim the initial left padding region
        gt_aggregated = gt_aggregated[window_size:]
        pred_aggregated = pred_aggregated[window_size:]

    else:
        # Non-ANN branch, all models below are dense models
        # Suppose dataset_eval has sample_counts keyed by path
        num_samples = len(dataset_eval)
        total_timesteps = args_dict["window_size"] + ((num_samples - 1) * stride)

        # Prepare placeholders for predictions
        # +1 to account for the mask class
        if model_choice == "any2any":
            num_classes = args_dict["num_classes"] + 1
        else:
            num_classes = args_dict["num_classes"]
        pred_matrix = np.full((num_samples, total_timesteps), -1, dtype=int)
        gt_matrix = np.full((num_samples, total_timesteps), -1, dtype=int)
        logits_matrix = np.full(
            (num_samples, total_timesteps, num_classes), -1, dtype=np.float32
        )

        dataloader_eval = DataLoader(
            dataset_eval,
            batch_size=eval_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=1,
        )

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader_eval):
                # Forward pass
                if model_choice == "any2any":
                    if dataset_eval.window_size == dataset_eval.inner_window_size:
                        (
                            emg_window,
                            action_window,
                            masked_emg,
                            masked_actions,
                            mask_positions_emg,
                            mask_positions_actions,
                            task_idx,
                            transition_index,
                            untokenized_emg,
                        ) = batch_data
                        emg_window = emg_window.to(device)
                        action_window = action_window.to(device)
                        masked_emg = masked_emg.to(device)
                        masked_actions = masked_actions.to(device)
                        mask_positions_emg = mask_positions_emg.to(device)
                        mask_positions_actions = mask_positions_actions.to(device)
                        task_idx = task_idx.to(device)
                        emg_output, action_output = model(
                            masked_emg, masked_actions, task_idx, mask_positions_emg
                        )
                    else:
                        (
                            emg_window,
                            coarse_action,
                            masked_coarse_actions,
                            mask_positions_coarse_emg,
                            mask_positions_coarse_actions,
                            task_idx,
                            transition_index,
                            untokenized_emg,
                            action_window,
                        ) = batch_data
                        emg_window = emg_window.to(device)
                        coarse_action = coarse_action.to(device)
                        masked_coarse_actions = masked_coarse_actions.to(device)
                        mask_positions_coarse_emg = mask_positions_coarse_emg.to(device)
                        mask_positions_coarse_actions = (
                            mask_positions_coarse_actions.to(device)
                        )
                        task_idx = task_idx.to(device)
                        transition_index = transition_index.to(device)
                        untokenized_emg = untokenized_emg.to(device)
                        action_window = action_window.to(device)

                        emg_output, action_output = model(
                            emg_window,
                            masked_coarse_actions,
                            task_idx,
                            mask_positions_coarse_emg,
                        )
                else:
                    (emg_window, action_window, raw_label_seq) = batch_data
                    emg_window = emg_window.to(device)
                    action_output = model(emg_window)

                if model_choice == "tra_hgr":
                    current_batch_size = emg_window.shape[0]
                    predicted_action_tokens = (
                        torch.argmax(action_output, dim=-1).cpu().numpy()
                    )
                    for i in range(current_batch_size):
                        global_idx = batch_idx * eval_batch_size + i
                        cur_index = global_idx * stride
                        if cur_index > total_timesteps:
                            end_idx = total_timesteps
                        pred_matrix[global_idx, cur_index] = predicted_action_tokens[i]
                        gt_matrix[global_idx, cur_index] = raw_label_seq[i, -1]
                        logits_matrix[global_idx, cur_index, :] = action_output[i]
                else:
                    # Upsample if resolution is lower
                    if action_output.size(1) < args_dict["window_size"]:
                        if (
                            model_choice == "any2any"
                            and dataset_eval.use_mav_for_emg == 0
                        ):
                            inner_window_size = args_dict["inner_window_size"]
                            inner_stride = args_dict["inner_window_size"]
                        elif (
                            model_choice == "any2any"
                            and dataset_eval.use_mav_for_emg == 1
                        ):
                            inner_window_size = args_dict["inner_window_size"]
                            inner_stride = args_dict["mav_inner_stride"]
                        elif model_choice == "ed_tcn":
                            inner_window_size = 150
                            inner_stride = 25
                        elif model_choice == "lstm":
                            inner_window_size = 100
                            inner_stride = 1
                        else:
                            raise Exception("model_choice not recognized")

                        action_output = repeat_chunks(
                            action_output,
                            args_dict["window_size"],
                            inner_window_size,
                            inner_stride,
                        )

                    # Obtain predicted labels, ground truth, and the logits
                    # Depending on the type of aggregation, realtime_online_aggregation() will choose to use pred_matrix or logits_matrix
                    predicted_action_tokens = (
                        torch.argmax(action_output, dim=-1).cpu().numpy()
                    )
                    action_output_vals = action_output.cpu().numpy()
                    if model_choice == "any2any":
                        ground_truth_action_tokens = action_window.cpu().numpy()
                    else:
                        ground_truth_action_tokens = raw_label_seq

                    # Fill in pred_matrix, gt_matrix, logits_matrix
                    current_batch_size = emg_window.shape[0]
                    for i in range(current_batch_size):
                        global_idx = batch_idx * eval_batch_size + i
                        start_idx = global_idx * stride
                        end_idx = start_idx + args_dict["window_size"]
                        if end_idx > total_timesteps:
                            end_idx = total_timesteps

                        pred_matrix[global_idx, start_idx:end_idx] = (
                            predicted_action_tokens[i, : (end_idx - start_idx)]
                        )
                        gt_matrix[global_idx, start_idx:end_idx] = (
                            ground_truth_action_tokens[i, : (end_idx - start_idx)]
                        )
                        logits_matrix[global_idx, start_idx:end_idx, :] = (
                            action_output_vals[i, : (end_idx - start_idx), :]
                        )

        gt_aggregated = build_gt_sequence(
            gt_matrix,
            stride,
            args_dict["window_size"],
            total_timesteps,
            lookahead,
            samples_between_prediction,
        )
        pred_aggregated = realtime_online_aggregation(
            pred_matrix=pred_matrix,
            logits_matrix=logits_matrix,
            window_size=args_dict["window_size"],
            stride=stride,
            lookahead=lookahead,
            samples_between_prediction=samples_between_prediction,
            maj_vote_range=maj_vote_range,
            likelihood_format=likelihood_format,
            weight_max_factor=weight_max_factor,
        )

        # Only trim if lookahead > 0
        if lookahead > 0:
            pred_aggregated = pred_aggregated[:-lookahead]
            gt_aggregated = gt_aggregated[:-lookahead]

    # Evaluate
    # Shared for both ANN and dense models
    results = evaluate_predictions(
        predicted_actions=pred_aggregated,
        ground_truth_actions=gt_aggregated,
        buffer_range=buffer_range,
        saved_checkpoint_pth=saved_checkpoint_pth,
        allow_relax=allow_relax,
    )

    # Plot (if verbose=1)
    plot_event_classification(
        path=csv_path,
        pred_sequence=pred_aggregated,
        gt_sequence=gt_aggregated,
        downsample=downsample,
        args_dict=args_dict,
        saved_checkpoint_pth=saved_checkpoint_pth,
        results=results,
        lookahead=lookahead,
        verbose=verbose,
    )

    print("\n** Current Accuracy **")
    print(f"  Transition Accuracy: {results['transition_window_accuracy']:.4f}")
    print(f"  Raw Accuracy:   {results['overall_accuracy']:.4f}")
    print(f"  Reasons:        {results['transition_reasons']}")
    print(f"  Transition (Event) Count:    {results['event_counts']}")
    print("--------------------------------------------------------\n")

    return results, pred_aggregated, gt_aggregated


def gather_csv_paths(files_or_dirs, args_dict):
    """
    Given a list of paths (files or directories), recursively gather all CSV files
    whose basename:
      1. contains no "movement" or "unlabeled",
      2. ends with ".csv",
      3. **starts with one of the strings in args_dict['val_patient_ids'] followed by an underscore**.
    Returns a list of matching CSV file paths.
    """
    val_patient_ids = args_dict["val_patient_ids"]
    all_csv_paths = []
    for path_item in files_or_dirs:
        if os.path.isdir(path_item):
            for root, dirs, files in os.walk(path_item):
                for f in files:
                    filename_lower = f.lower()
                    if filename_lower.endswith(".csv"):
                        if (
                            "movement" in filename_lower
                            or "unlabeled" in filename_lower
                        ):
                            continue
                        if any(f.startswith(pid + "_") for pid in val_patient_ids):
                            all_csv_paths.append(os.path.join(root, f))
        elif os.path.isfile(path_item):
            f = os.path.basename(path_item)
            filename_lower = f.lower()
            if filename_lower.endswith(".csv"):
                if "movement" in filename_lower or "unlabeled" in filename_lower:
                    continue
                if any(f.startswith(pid + "_") for pid in val_patient_ids):
                    all_csv_paths.append(path_item)
        else:
            print(f"Skipping invalid path: {path_item}")
    return all_csv_paths


def gather_epn_paths(epn_data_master_folder):
    """
    Example: gather .npy from subdirectories named 'testingJSON_user*'
    """
    if not os.path.isdir(epn_data_master_folder):
        raise ValueError(f"EPN data folder does not exist: {epn_data_master_folder}")

    epn_file_paths = []
    for subject_folder in os.listdir(epn_data_master_folder):
        subject_folder_path = os.path.join(epn_data_master_folder, subject_folder)
        if not os.path.isdir(subject_folder_path):
            continue
        if not subject_folder.startswith("testingJSON_user"):
            continue
        for fname in os.listdir(subject_folder_path):
            if fname.lower().endswith(".npy"):
                full_path = os.path.join(subject_folder_path, fname)
                epn_file_paths.append(full_path)
    return epn_file_paths


###############################################################################
#                                MAIN
###############################################################################


def main(
    saved_checkpoint_pth,
    eval_batch_size,
    eval_task,
    transition_samples_only,
    buffer_range,
    mask_percentage,
    mask_type,
    stride,
    files_or_dirs,
    allow_relax,
    lookahead,
    weight_max_factor,
    likelihood_format,
    samples_between_prediction,
    maj_vote_range,
    epn_eval,
    recog_threshold,
    verbose,
    model_choice,
    sample_range=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    if saved_checkpoint_pth is not None:
        print("Loading pretrained weights...")
        checkpoint = torch.load(
            saved_checkpoint_pth, map_location=device, weights_only=False
        )
        args_dict = checkpoint["args_dict"]
    else:
        raise Exception("No pretrained model path specified")

    checkpoint_identifier = build_checkpoint_identifier(saved_checkpoint_pth)

    torch.manual_seed(args_dict["seed"])
    rdm.seed(args_dict["seed"])
    np.random.seed(args_dict["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Depending on epn_eval, gather files:
    if epn_eval == 1:
        epn_data_master_folder = files_or_dirs[0]
        csv_paths_all = gather_epn_paths(epn_data_master_folder)
    else:
        csv_paths_all = gather_csv_paths(files_or_dirs, args_dict)

    if not csv_paths_all:
        raise ValueError("No valid files found for evaluation.")

    # Filter out files, e.g. 'trainingJSON_user' or 'unlabeled' ...
    csv_paths_all = [
        p
        for p in csv_paths_all
        if "trainingJSON_user" not in p.lower() and "unlabeled" not in p.lower()
    ]
    if args_dict["num_classes"] == 3:
        csv_paths_all = [
            p
            for p in csv_paths_all
            if not any(
                substring in os.path.basename(p).lower()
                for substring in ["wavein", "waveout", "pinch"]
            )
        ]

    # Initialize accumulators
    total_events = 0
    total_correct_events = 0.0
    sum_raw_accuracies = 0.0

    # storing std
    event_accuracies_per_file = []
    raw_accuracies_per_file = []

    # subject-level statistics
    if epn_eval == 1:
        subj_event_dict = defaultdict(list)
        subj_raw_dict = defaultdict(list)

    # Will store per-file results
    results_across_files = []

    # gather transition reasons here on-the-fly
    all_reasons = []

    # initialize model
    model = initialize_model(args_dict, checkpoint, model_choice, device)

    # sort csv list before loop to ensure same ordering on different machines
    csv_paths_all.sort()

    if sample_range is not None:
        start_idx, end_idx = sample_range
        csv_paths_all = csv_paths_all[start_idx:end_idx]
        print(
            f"Slicing csv_paths_all to indices [{start_idx}:{end_idx}]. "
            f"Number of files now = {len(csv_paths_all)}"
        )
        if not csv_paths_all:
            raise ValueError("No valid files remain after applying --sample_range.")

    # Evaluate each file
    for i, csv_path in enumerate(csv_paths_all):
        print(f"Processing file {i + 1}/{len(csv_paths_all)}: {csv_path}")

        dataset_eval = initialize_dataset(
            model_choice,
            args_dict,
            csv_path,
            stride,
            eval_task,
            transition_samples_only,
            mask_percentage,
            mask_type,
        )

        # Check if this dataset has any samples at all (if the entire sample is shorter than window_size)
        if len(dataset_eval) == 0:
            print(
                f"Skipping file (too few samples for a window of size {args_dict['window_size']}): {csv_path}"
            )
            continue  # move on to the next file

        results, pred_seq, gt_seq = process_and_evaluate(
            model=model,
            dataset_eval=dataset_eval,
            device=device,
            eval_batch_size=eval_batch_size,
            eval_task=eval_task,
            csv_path=csv_path,
            downsample=False,
            args_dict=args_dict,
            buffer_range=buffer_range,
            saved_checkpoint_pth=saved_checkpoint_pth,
            allow_relax=allow_relax,
            lookahead=lookahead,
            weight_max_factor=weight_max_factor,
            likelihood_format=likelihood_format,
            samples_between_prediction=samples_between_prediction,
            maj_vote_range=maj_vote_range,
            stride=stride,
            epn_eval=epn_eval,
            recog_threshold=recog_threshold,
            verbose=verbose,
            model_choice=model_choice,
        )

        # Store this file's results
        results_across_files.append((csv_path, results, pred_seq, gt_seq))

        # -------------------------------------
        # Update partial sums
        # -------------------------------------
        event_acc = results["transition_window_accuracy"]
        event_counts = results["event_counts"]
        raw_acc = results["overall_accuracy"]

        # Append to the per-file lists:
        event_accuracies_per_file.append(event_acc)
        raw_accuracies_per_file.append(raw_acc)

        # Extract subject id from the file path"../data/EMG-EPN-612/testingJSON_user<id>/<file>.npy"
        if epn_eval == 1:
            m = re.search(r"testingJSON_user(\d+)", csv_path)
            subj_id = m.group(1) if m else "unknown"
            subj_event_dict[subj_id].append(event_acc)
            subj_raw_dict[subj_id].append(raw_acc)

        total_correct_events += event_acc * event_counts
        total_events += event_counts
        sum_raw_accuracies += raw_acc

        # Add this file's transition reasons to our global list
        all_reasons.extend(results["transition_reasons"])

        # Compute partial (cumulative) results for printing
        partial_event_accuracy = (
            (total_correct_events / total_events) if total_events > 0 else 0.0
        )
        partial_raw_accuracy = sum_raw_accuracies / (i + 1)
        partial_event_std = np.std(event_accuracies_per_file)
        partial_raw_std = np.std(raw_accuracies_per_file)

        # Print partial (cumulative) results
        print(f"\n** Cumulative Results After File {i + 1}/{len(csv_paths_all)} **")
        print(
            f"  Transition Accuracy so far: {partial_event_accuracy:.4f} ± {partial_event_std:.4f}"
        )
        print(
            f"  Raw Accuracy so far:   {partial_raw_accuracy:.4f} ± {partial_raw_std:.4f}"
        )

        print("--------------------------------------------------------\n")
    # ------------------------------------------------
    # After processing all files, compute final metrics
    # ------------------------------------------------
    num_files = len(results_across_files)
    avg_event_accuracy = (
        (total_correct_events / total_events) if total_events > 0 else 0.0
    )
    avg_raw_accuracy = (sum_raw_accuracies / num_files) if num_files > 0 else 0.0

    # Standard deviations
    std_event_accuracy = (
        np.std(event_accuracies_per_file) if event_accuracies_per_file else 0.0
    )
    std_raw_accuracy = (
        np.std(raw_accuracies_per_file) if raw_accuracies_per_file else 0.0
    )

    # Collapse multiple files of the same subject into one mean
    if epn_eval == 1 and subj_event_dict:
        subj_event_means = [np.mean(v) for v in subj_event_dict.values()]
        subj_raw_means = [np.mean(v) for v in subj_raw_dict.values()]
        mean_event_subj = np.mean(subj_event_means)
        std_event_subj = np.std(subj_event_means)
        mean_raw_subj = np.mean(subj_raw_means)
        std_raw_subj = np.std(subj_raw_means)
    else:
        mean_event_subj = std_event_subj = mean_raw_subj = std_raw_subj = None

    # Build a counter over all transition reasons
    reason_counter = Counter(all_reasons)

    epn_eval_conditioning = "epn_eval" if epn_eval == 1 else "roam_eval"

    folder_name = f"{checkpoint_identifier}_LA{lookahead}"
    summary_dir = os.path.join("output", folder_name)
    os.makedirs(summary_dir, exist_ok=True)

    # Build a little range string, e.g. "_range_0-5" if sample_range=(0,5)
    # or just "" if sample_range=None
    range_str = ""
    if sample_range is not None:
        start_idx, end_idx = sample_range
        range_str = f"_range_{start_idx}-{end_idx}"

    # Add both lookahead and range to the filename
    summary_filename = (
        f"evaluation_summary_{epn_eval_conditioning}_"
        f"{checkpoint_identifier}_LA{lookahead}{range_str}.txt"
    )
    summary_path = os.path.join(summary_dir, summary_filename)

    with open(summary_path, "w") as f:
        f.write("=== Overall Summary ===\n")
        f.write(
            f"Transition Accuracy (weighted) (avg ± std): "
            f"{avg_event_accuracy:.4f} ± {std_event_accuracy:.4f}\n"
        )
        f.write(
            f"Raw Accuracy (average) (avg ± std): "
            f"{avg_raw_accuracy:.4f} ± {std_raw_accuracy:.4f}\n"
        )

        if epn_eval == 1 and mean_event_subj is not None:
            f.write(f"\n--- Per‑Subject Statistics (N={len(subj_event_means)}) ---\n")
            f.write(
                f"Transition Accuracy (avg ± std): "
                f"{mean_event_subj:.4f} ± {std_event_subj:.4f}\n"
            )
            f.write(
                f"Raw Accuracy (avg ± std): {mean_raw_subj:.4f} ± {std_raw_subj:.4f}\n"
            )

        f.write(f"Total Transition (Event): {total_events}\n\n")

        f.write("=== Distribution of Transition Reasons ===\n")
        for reason, count in reason_counter.items():
            f.write(f"   {reason}: {count}\n")
        f.write("\nEnd of summary.\n")

    print(f"Final summary saved to: {summary_path}")

    if epn_eval == 1 and mean_event_subj is not None:
        print("\n=== Per‑Subject Summary ===")
        print(
            f"  Transition Acc: {mean_event_subj:.4f} ± {std_event_subj:.4f} "
            f"(N={len(subj_event_means)} subjects)"
        )
        print(
            f"  Raw Acc:        {mean_raw_subj:.4f} ± {std_raw_subj:.4f} "
            f"(N={len(subj_raw_means)} subjects)"
        )

    # Write JSON with per-file details if verbose=1
    if verbose == 1:
        details_filename = (
            f"evaluation_details_{epn_eval_conditioning}_{checkpoint_identifier}.json"
        )
        details_path = os.path.join(summary_dir, details_filename)

        details_dict = {}
        for csv_path, file_results, pred_seq, gt_seq in results_across_files:
            pred_seq_list = (
                pred_seq.tolist()
                if isinstance(pred_seq, np.ndarray)
                else list(pred_seq)
            )
            gt_seq_list = (
                gt_seq.tolist() if isinstance(gt_seq, np.ndarray) else list(gt_seq)
            )

            file_info = {
                "transition_window_accuracy": file_results[
                    "transition_window_accuracy"
                ],
                "event_count": file_results["event_counts"],
                "raw_accuracy": file_results["overall_accuracy"],
                "transition_reasons": file_results["transition_reasons"],
                "pred_seq": pred_seq_list,
                "gt_seq": gt_seq_list,
            }

            details_dict[csv_path] = file_info

        with open(details_path, "w") as jf:
            json.dump(details_dict, jf, indent=4)

        print(f"File-level details JSON saved to: {details_path}")

    return avg_event_accuracy, avg_raw_accuracy


###############################################################################
#                               ENTRY POINT
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--saved_checkpoint_pth",
        default=None,
        type=str,
        help="Path to pretrained model checkpoint.",
    )
    parser.add_argument("--eval_batch_size", default=64, type=int)
    parser.add_argument("--eval_task", required=True, type=str)
    parser.add_argument("--transition_samples_only", action="store_true")
    parser.add_argument("--buffer_range", required=True, type=int)
    parser.add_argument("--mask_percentage", default=0.6, type=float)
    parser.add_argument("--mask_type", default="poisson", type=str)
    parser.add_argument("--stride", default=1, type=int)
    parser.add_argument("--files_or_dirs", nargs="+", required=True)
    parser.add_argument("--allow_relax", default=0, type=int, choices=[0, 1])
    parser.add_argument("--lookahead", type=int, default=0)
    parser.add_argument("--weight_max_factor", type=float, default=1.0)
    parser.add_argument(
        "--likelihood_format",
        type=str,
        default="logits",
        choices=["logits", "probs", "argmax"],
    )
    parser.add_argument("--samples_between_prediction", type=int, default=1)
    parser.add_argument(
        "--maj_vote_range", type=str, default="single", choices=["single", "future"]
    )
    parser.add_argument(
        "--epn_eval",
        default=0,
        type=int,
        choices=[0, 1],
        help="If 1, use EPN-based .npy files and compute EPN metrics.",
    )
    parser.add_argument("--recog_threshold", default=0.5, type=float)
    parser.add_argument(
        "--verbose",
        default=0,
        type=int,
        choices=[0, 1],
        help="If 1, generate plots & per-file JSON details; if 0, skip them.",
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        required=True,
        help="Select which model to use for evaluation",
    )
    parser.add_argument(
        "--sample_range",
        nargs=2,
        type=int,
        default=None,
        help=(
            "If specified, slice the sorted file list to only these indices "
            "[start_idx, end_idx). For distributing evaluation across multiple GPUs."
        ),
    )
    args = parser.parse_args()

    main(
        args.saved_checkpoint_pth,
        args.eval_batch_size,
        args.eval_task,
        args.transition_samples_only,
        args.buffer_range,
        args.mask_percentage,
        args.mask_type,
        args.stride,
        args.files_or_dirs,
        args.allow_relax,
        args.lookahead,
        args.weight_max_factor,
        args.likelihood_format,
        args.samples_between_prediction,
        args.maj_vote_range,
        args.epn_eval,
        args.recog_threshold,
        args.verbose,
        args.model_choice,
        args.sample_range,
    )
