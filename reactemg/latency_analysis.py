#!/usr/bin/env python3
"""
Latency Analysis for EMG Event Classification

This script analyzes the temporal offset (latency) between predicted transitions
and ground truth transitions for successful/correct predictions.

For each successful transition, it measures how many timesteps away the predicted
transition occurs from the ground truth transition within a buffer range.
"""

import argparse
import json
import numpy as np
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


def find_transition_indices(sequence):
    """
    Find indices where transitions occur in a sequence.

    Parameters
    ----------
    sequence : list or np.ndarray
        A sequence of integer labels

    Returns
    -------
    transition_indices : list of int
        Indices where the label changes from sequence[i] to sequence[i+1]
    """
    if len(sequence) <= 1:
        return []

    sequence = np.array(sequence)
    # Find where sequence[i] != sequence[i+1]
    transitions = np.where(sequence[:-1] != sequence[1:])[0]
    # The transition happens between index i and i+1, we return i
    return transitions.tolist()


def find_predicted_transition_in_buffer(pred_seq, gt_seq, gt_transition_idx,
                                       buffer_range, old_class, new_class):
    """
    Find the predicted transition from old_class to new_class within a buffer
    range around the ground truth transition.

    Parameters
    ----------
    pred_seq : np.ndarray
        Full predicted sequence
    gt_seq : np.ndarray
        Full ground truth sequence
    gt_transition_idx : int
        Index of the ground truth transition (where gt_seq[i] != gt_seq[i+1])
    buffer_range : int
        Total buffer range (will search ±buffer_range//2 around gt_transition_idx)
    old_class : int
        The class before the transition
    new_class : int
        The class after the transition

    Returns
    -------
    signed_latency : int or None
        Signed distance (in timesteps) from predicted transition to ground truth.
        Negative values indicate prediction before GT, positive after GT.
        Returns None if no valid transition found in buffer.
    """
    half_range = buffer_range // 2

    # Define search window
    start = max(0, gt_transition_idx - half_range)
    end = min(len(pred_seq) - 1, gt_transition_idx + half_range)

    # Look for transitions in the predicted sequence within this range
    # A transition from old_class to new_class occurs at index i if:
    #   pred_seq[i] == old_class and pred_seq[i+1] == new_class

    candidates = []
    for i in range(start, end):
        if i + 1 < len(pred_seq):
            if pred_seq[i] == old_class and pred_seq[i + 1] == new_class:
                # Found a transition at index i
                signed_distance = i - gt_transition_idx
                abs_distance = abs(signed_distance)
                candidates.append((abs_distance, signed_distance, i))

    # If no exact transition found, look for first occurrence of new_class
    # This handles cases where prediction might already be at new_class
    if not candidates:
        for i in range(start, end + 1):
            if i < len(pred_seq) and pred_seq[i] == new_class:
                # Check if this is after being in old_class or at boundary
                # Take the first occurrence of new_class
                signed_distance = i - gt_transition_idx
                abs_distance = abs(signed_distance)
                candidates.append((abs_distance, signed_distance, i))
                break

    if not candidates:
        return None

    # Return the closest one (minimum absolute distance)
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]  # Return signed distance


def analyze_latency(json_path, buffer_range=200):
    """
    Analyze latency for all successful transitions in the JSON file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file produced by event_classification.py with --verbose 1
    buffer_range : int
        Buffer range to search for predicted transitions (default: 200)

    Returns
    -------
    results : dict
        Dictionary containing latency distribution and statistics
    """
    print(f"Loading JSON from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Initialize counters
    latency_values = []
    latency_distribution = {str(i): 0 for i in range(buffer_range // 2 + 1)}

    total_transitions_across_files = 0
    correct_transitions_across_files = 0
    transitions_with_latency_found = 0
    transitions_without_latency = 0

    print(f"\nProcessing {len(data)} files...")

    # Iterate through each file in the JSON
    for file_path, file_info in data.items():
        pred_seq = np.array(file_info['pred_seq'])
        gt_seq = np.array(file_info['gt_seq'])
        transition_reasons = file_info['transition_reasons']

        # Find ground truth transitions
        gt_transition_indices = find_transition_indices(gt_seq)

        # Verify that the number of transitions matches the reasons list
        num_transitions = len(gt_transition_indices)
        if len(transition_reasons) != num_transitions:
            print(f"Warning: Mismatch in {file_path}")
            print(f"  GT transitions: {num_transitions}, Reasons: {len(transition_reasons)}")
            # Use the minimum to avoid index errors
            num_transitions = min(num_transitions, len(transition_reasons))

        total_transitions_across_files += num_transitions

        # Process each transition
        for i in range(num_transitions):
            reason = transition_reasons[i]

            # Only process successful transitions
            if reason == "Successful":
                correct_transitions_across_files += 1

                gt_trans_idx = gt_transition_indices[i]

                # Determine old and new classes
                old_class = gt_seq[gt_trans_idx]
                new_class = gt_seq[gt_trans_idx + 1] if gt_trans_idx + 1 < len(gt_seq) else old_class

                # Find predicted transition within buffer
                signed_latency = find_predicted_transition_in_buffer(
                    pred_seq, gt_seq, gt_trans_idx, buffer_range, old_class, new_class
                )

                if signed_latency is not None:
                    abs_latency = abs(signed_latency)
                    latency_values.append(abs_latency)
                    transitions_with_latency_found += 1

                    # Update absolute distribution (clamp to max range)
                    dist_key = str(min(abs_latency, buffer_range // 2))
                    latency_distribution[dist_key] += 1
                else:
                    transitions_without_latency += 1

    # Compute statistics
    if latency_values:
        average_latency = float(np.mean(latency_values))
        std_latency = float(np.std(latency_values, ddof=1 if len(latency_values) > 1 else 0))
        median_latency = float(np.median(latency_values))
        min_latency = int(np.min(latency_values))
        max_latency = int(np.max(latency_values))
    else:
        average_latency = 0.0
        std_latency = 0.0
        median_latency = 0.0
        min_latency = 0
        max_latency = 0

    # Build results dictionary
    results = {
        "buffer_range": buffer_range,
        "total_transitions": total_transitions_across_files,
        "correct_transitions": correct_transitions_across_files,
        "transitions_with_latency_found": transitions_with_latency_found,
        "transitions_without_latency": transitions_without_latency,
        "average_latency": average_latency,
        "std_latency": std_latency,
        "median_latency": median_latency,
        "min_latency": min_latency,
        "max_latency": max_latency,
        "latency_distribution": latency_distribution,
        "raw_latency_values": latency_values
    }

    return results


def print_summary(results):
    """Print a human-readable summary of the latency analysis."""
    print("\n" + "="*70)
    print("LATENCY ANALYSIS SUMMARY")
    print("="*70)
    print(f"Buffer Range: {results['buffer_range']} (±{results['buffer_range']//2} timesteps)")
    print(f"\nTransition Counts:")
    print(f"  Total transitions: {results['total_transitions']}")
    print(f"  Correct/Successful transitions: {results['correct_transitions']}")
    print(f"  Transitions with latency found: {results['transitions_with_latency_found']}")
    print(f"  Transitions without latency: {results['transitions_without_latency']}")

    if results['transitions_with_latency_found'] > 0:
        print(f"\nLatency Statistics (in timesteps):")
        print(f"  Average: {results['average_latency']:.2f}")
        print(f"  Std Dev: {results['std_latency']:.2f}")
        print(f"  Median:  {results['median_latency']:.2f}")
        print(f"  Min:     {results['min_latency']}")
        print(f"  Max:     {results['max_latency']}")

        # Show top latency bins
        print(f"\nTop 10 Latency Bins (distance: count):")
        dist_items = [(int(k), v) for k, v in results['latency_distribution'].items() if v > 0]
        dist_items.sort(key=lambda x: x[1], reverse=True)
        for dist, count in dist_items[:10]:
            pct = 100 * count / results['transitions_with_latency_found']
            print(f"  {dist:3d} timesteps: {count:4d} ({pct:5.1f}%)")
    else:
        print("\nNo latency measurements found!")

    print("="*70 + "\n")


def plot_latency_histogram(results, output_path):
    """
    Generate and save a histogram of latency distribution (absolute distance).

    Parameters
    ----------
    results : dict
        Results dictionary from analyze_latency()
    output_path : Path or str
        Path to save the histogram PNG file
    """
    if results['transitions_with_latency_found'] == 0:
        print("No latency data to plot.")
        return

    # Prepare data for histogram (0 to 100 timesteps)
    max_distance = 100
    distances = list(range(max_distance + 1))
    counts = []

    for dist in distances:
        dist_str = str(min(dist, results['buffer_range'] // 2))
        count = results['latency_distribution'].get(dist_str, 0)
        counts.append(count)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.bar(distances, counts, width=1.0, edgecolor='black', linewidth=0.5)

    plt.xlabel('Latency (timesteps)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Prediction Latency for Successful Transitions', fontsize=14)
    plt.grid(axis='y', alpha=0.3, linestyle='--')

    # Add statistics text box
    stats_text = (
        f'Mean: {results["average_latency"]:.2f}\n'
        f'Std Dev: {results["std_latency"]:.2f}\n'
        f'Median: {results["median_latency"]:.2f}\n'
        f'N: {results["transitions_with_latency_found"]}'
    )
    plt.text(0.98, 0.98, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)

    plt.xlim(-1, max_distance + 1)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Absolute latency histogram saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze latency of predicted transitions relative to ground truth."
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON file produced by event_classification.py (with --verbose 1)"
    )
    parser.add_argument(
        "--buffer_range",
        type=int,
        default=200,
        help="Buffer range to search for predicted transitions (default: 200, ±100 timesteps)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save output JSON (default: same directory as input with _latency_analysis.json suffix)"
    )
    parser.add_argument(
        "--save_histogram",
        action="store_true",
        help="Save a histogram plot of the latency distribution (PNG format)"
    )

    args = parser.parse_args()

    # Perform analysis
    results = analyze_latency(args.json_path, args.buffer_range)

    # Print summary
    print_summary(results)

    # Determine output path
    if args.output_path is None:
        input_path = Path(args.json_path)
        output_path = input_path.parent / (input_path.stem + "_latency_analysis.json")
    else:
        output_path = Path(args.output_path)

    # Save results
    # Remove raw values for cleaner JSON (optional)
    results_to_save = results.copy()
    raw_values = results_to_save.pop('raw_latency_values', [])

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)

    print(f"Latency analysis saved to: {output_path}")

    # Optionally save raw values separately
    if raw_values:
        raw_output_path = output_path.parent / (output_path.stem + "_raw_values.json")
        with open(raw_output_path, 'w') as f:
            json.dump({"raw_latency_values": raw_values}, f, indent=4)
        print(f"Raw latency values saved to: {raw_output_path}")

    # Generate histogram if requested
    if args.save_histogram:
        # Absolute latency histogram
        histogram_path = output_path.parent / (output_path.stem + "_histogram.png")
        plot_latency_histogram(results, histogram_path)


if __name__ == "__main__":
    main()
