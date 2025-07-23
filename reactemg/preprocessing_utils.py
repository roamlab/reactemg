import argparse
import random
import os
import glob
import math
from nn_models import (
    Any2Any_Model,
    EDTCN_Model,
    LSTM_Model,
    ANN_Model,
)
from dataset import (
    Any2Any_Dataset,
    EDTCN_Dataset,
    LSTM_Dataset,
    ANN_Dataset,
)


def parse_tuple(s):
    try:
        return tuple(map(float, s.strip("()").split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("Tuples must be in the form x,y")


def get_csv_paths(
    dataset_selection,
    num_classes,
    roam_data_master_folder,
    roam_data_subfolders,
    public_data_folders,
    epn_data_master_folder,
    val_patient_ids,
    epn_subset_percentage,
    discard_labeled_percentage,
):
    """
    Construct dataset paths

    Parameters
    ----------
    dataset_selection : str
        Which dataset combination is selected:
          - roam_only
          - pub_only
          - pub_with_roam
          - epn_only
          - pub_with_roam_with_epn
          - pub_with_epn

    num_classes : int
        Number of classes (affects which EPN files are skipped).
    roam_data_master_folder : str
        Path to the main ROAM dataset folder.
    roam_data_subfolders : list of str
        Subfolders within the ROAM dataset that contain CSVs.
    public_data_folders : list of str
        Folders for the public datasets.
    epn_data_master_folder : str
        Path to the main EPN dataset folder.
    val_patient_ids : list of str, optional
        List of two-letter patient IDs for the ROAM validation set.
    epn_subset_percentage : float, optional
        Fraction of EPN files (0.0 - 1.0) to keep from the eligible set if
        dataset_selection == "pub_with_roam_with_epn" OR == "pub_with_epn".

    Returns
    -------
    train_paths : list of str
        Paths to selected training files.
    val_paths : list of str
        Paths to selected validation files.
    """

    if val_patient_ids is None:
        val_patient_ids = []

    # =========================================================
    # MODE: "pub_with_epn"
    #   - Combine public + EPN only
    #   - Apply "pub_with_roam_with_epn"-style EPN logic:
    #       (skip wave/pinch if num_classes==3, keep only open/fist, downsample)
    #   - Then 5% of combined => val, rest => train.
    # =========================================================
    if dataset_selection in ["pub_with_epn", "epn_only"]:
        # 1) Gather Public data
        pub_paths = []
        if dataset_selection in ["pub_with_epn"]:
            for pub_folder in public_data_folders:
                csv_files = glob.glob(
                    os.path.join(pub_folder, "**", "*.csv"), recursive=True
                )
                pub_paths.extend(csv_files)

        # 2) Gather EPN data
        if not os.path.isdir(epn_data_master_folder):
            raise ValueError(
                f"EPN data folder does not exist: {epn_data_master_folder}"
            )

        epn_file_paths = []
        for subject_folder in os.listdir(epn_data_master_folder):
            subject_folder_path = os.path.join(epn_data_master_folder, subject_folder)

            # Skip non-directories or "testingJSON_user..." folders
            if not os.path.isdir(subject_folder_path):
                continue
            if subject_folder.startswith("testingJSON_user"):
                continue

            # Gather the .npy files
            for fname in os.listdir(subject_folder_path):
                if not fname.lower().endswith(".npy"):
                    continue

                fname_lc = fname.lower()
                full_path = os.path.join(subject_folder_path, fname)

                # Skip wave/pinch if num_classes==3
                if num_classes == 3:
                    skip_substrings = ["wavein", "waveout", "pinch"]
                    if any(sub in fname_lc for sub in skip_substrings):
                        continue

                epn_file_paths.append(full_path)

        # When we have the public dataset, keep only "open" or "fist" (same logic as "pub_with_roam_with_epn")
        if dataset_selection in ["pub_with_epn"]:
            epn_file_paths = [
                p
                for p in epn_file_paths
                if (
                    "open" in os.path.basename(p).lower()
                    or "fist" in os.path.basename(p).lower()
                )
            ]

        # Downsample EPN to epn_subset_percentage
        random.shuffle(epn_file_paths)
        subset_size = int(len(epn_file_paths) * epn_subset_percentage)
        epn_file_paths = epn_file_paths[:subset_size]

        # Combine Public + EPN
        combined_paths = pub_paths + epn_file_paths
        random.shuffle(combined_paths)

        # 3) Split 5% for validation
        val_size = int(0.05 * len(combined_paths))
        val_paths = combined_paths[:val_size]
        train_paths = combined_paths[val_size:]

    else:
        # ================================
        # For all other modes, same logic
        # ================================
        train_paths = []
        val_paths = []

        # -----------------------------------------
        # 1) Process ROAM data
        # -----------------------------------------
        # We will always gather validation data from ROAM using val_patient_ids,
        # but only add to the training set if dataset_selection uses ROAM data.
        for folder in roam_data_subfolders:
            folder_path = os.path.join(roam_data_master_folder, folder)

            # Recursively find all .csv files
            csv_files = glob.glob(
                os.path.join(folder_path, "**", "*.csv"), recursive=True
            )

            for csv_file in csv_files:
                basename = os.path.basename(csv_file)
                root, _ = os.path.splitext(basename)
                patient_id = root.split("_", 1)[0]

                if patient_id in val_patient_ids:
                    val_paths.append(csv_file)
                elif dataset_selection in [
                    "roam_only",
                    "pub_with_roam",
                    "pub_with_roam_with_epn",
                ]:
                    train_paths.append(csv_file)

        # -----------------------------------------
        # 2) Process Public data
        # -----------------------------------------
        if dataset_selection in ["pub_only", "pub_with_roam", "pub_with_roam_with_epn"]:
            # Add any CSV from the public datasets to TRAIN set
            for pub_folder in public_data_folders:
                csv_files = glob.glob(
                    os.path.join(pub_folder, "**", "*.csv"), recursive=True
                )
                train_paths.extend(csv_files)

        # -----------------------------------------
        # 3) Process EPN data (if requested)
        # -----------------------------------------
        # 'epn_only' => use all EPN files, skipping wave/pinch if needed
        # 'pub_with_roam_with_epn' => same skipping logic, and sample epn_subset_percentage.
        if dataset_selection in ["pub_with_roam_with_epn"]:
            if not os.path.isdir(epn_data_master_folder):
                raise ValueError(
                    f"EPN data folder does not exist: {epn_data_master_folder}"
                )

            # Collect all valid EPN .npy file paths here, then sample if needed
            epn_file_paths = []

            for subject_folder in os.listdir(epn_data_master_folder):
                subject_folder_path = os.path.join(
                    epn_data_master_folder, subject_folder
                )

                # Skip anything that is not a directory
                if not os.path.isdir(subject_folder_path):
                    continue

                # Skip "testingJSON_user..." folders entirely
                if subject_folder.startswith("testingJSON_user"):
                    continue

                # Ensure no deeper nesting
                for entry in os.listdir(subject_folder_path):
                    sub_sub_path = os.path.join(subject_folder_path, entry)
                    if os.path.isdir(sub_sub_path):
                        raise ValueError(
                            f"Unexpected folder '{sub_sub_path}' under subject folder "
                            f"'{subject_folder_path}'. Only one level of nesting is allowed."
                        )

                # Gather the .npy files in this subject folder
                for fname in os.listdir(subject_folder_path):
                    if not fname.lower().endswith(".npy"):
                        continue

                    fname_lc = fname.lower()
                    full_path = os.path.join(subject_folder_path, fname)

                    # Skip logic depending on num_classes
                    if num_classes == 3:
                        # skip wavein, waveout, pinch
                        skip_substrings = ["wavein", "waveout", "pinch"]
                        if any(sub in fname_lc for sub in skip_substrings):
                            continue

                    epn_file_paths.append(full_path)

            if dataset_selection == "pub_with_roam_with_epn":
                # Keep only 'open' or 'fist' in filename
                epn_file_paths = [
                    p
                    for p in epn_file_paths
                    if (
                        "open" in os.path.basename(p).lower()
                        or "fist" in os.path.basename(p).lower()
                    )
                ]

                # Randomly sample epn_subset_percentage
                random.shuffle(epn_file_paths)
                subset_size = int(len(epn_file_paths) * epn_subset_percentage)
                epn_file_paths = epn_file_paths[:subset_size]

            # Add these EPN paths to train
            train_paths.extend(epn_file_paths)

    # Sorting then discarding (to ensure reproducibility)
    train_paths.sort()
    val_paths.sort()
    if discard_labeled_percentage > 0.0:
        # shuffle and discard
        random.shuffle(train_paths)
        num_to_discard = int(discard_labeled_percentage * len(train_paths))
        train_paths = train_paths[num_to_discard:]
        train_paths.sort()  # re-sort to show in ascending order (easy to debug)

    return train_paths, val_paths


def get_unlabeled_csv_paths(
    unlabeled_data_folder,
    labeled_paths_train,
    labeled_paths_val,
    epn_unlabeled_classes,
    unlabeled_percentage,
):
    """
    Gathers and returns unlabeled file paths, applying:
      1) EPN-based skip logic if epn_unlabeled_classes=3 (skips wave/pinch).
      2) Random sampling of unlabeled data by unlabeled_percentage (file-level).
      3) Removal of any files also appearing in labeled_paths_train or labeled_paths_val.
    Reproducibility guaranteed by sorting + seeding the shuffle.

    Parameters
    ----------
    unlabeled_data_folder : str
        Root folder holding unlabeled .csv or .npy files (EPN or other public).
    labeled_paths_train : list of str
        Already-final labeled training file paths (after discarding).
    labeled_paths_val : list of str
        Labeled validation file paths.
    epn_unlabeled_classes : int
        3 or 6. If 3 => skip wave/pinch from EPN folder. If 6 => keep them.
        (Does NOT depend on the model's num_classes!)
    unlabeled_percentage : float
        Fraction of unlabeled data to retain (0.0 => none, 1.0 => all).

    Returns
    -------
    final_unlabeled_paths : list of str
        The final subset of unlabeled file paths.
    """

    # 1) Gather all .csv/.npy from unlabeled_data_folder
    all_unlabeled_paths = []
    for root, dirs, files in os.walk(unlabeled_data_folder):
        # Sort to ensure consistent ordering across machines
        files_sorted = sorted(files)
        for fname in files_sorted:
            lower_name = fname.lower()
            if lower_name.endswith(".csv") or lower_name.endswith(".npy"):
                full_path = os.path.join(root, fname)

                # 2) If EPN folder => apply skip logic for 3-class
                # (only skip wave/pinch if epn_unlabeled_classes == 3)
                if "epn" in root.lower():
                    if epn_unlabeled_classes == 3:
                        skip_substrings = ["wavein", "waveout", "pinch"]
                        if any(sub in lower_name for sub in skip_substrings):
                            continue

                # For non-EPN folders => no gesture skipping
                # (public unlabeled data is always kept)
                all_unlabeled_paths.append(full_path)

    # 3) Remove collisions with labeled train/val
    labeled_basenames = set(
        [os.path.basename(p) for p in labeled_paths_train + labeled_paths_val]
    )

    cleaned_unlabeled = []
    for path_ in all_unlabeled_paths:
        if os.path.basename(path_) not in labeled_basenames:
            cleaned_unlabeled.append(path_)
    print(
        f"[get_unlabeled_csv_paths] Found {len(all_unlabeled_paths)} raw unlabeled files."
    )
    print(
        f"[get_unlabeled_csv_paths] After removing collisions: {len(cleaned_unlabeled)}"
    )

    # 4) Randomly sub-sample by unlabeled_percentage
    cleaned_unlabeled = sorted(cleaned_unlabeled)
    if unlabeled_percentage <= 0.0:
        final_unlabeled_paths = []
    elif unlabeled_percentage >= 1.0:
        final_unlabeled_paths = cleaned_unlabeled
    else:
        num_to_keep = int(len(cleaned_unlabeled) * unlabeled_percentage)
        random.shuffle(cleaned_unlabeled)
        final_unlabeled_paths = cleaned_unlabeled[:num_to_keep]
        final_unlabeled_paths = sorted(final_unlabeled_paths)

    print(
        f"[get_unlabeled_csv_paths] Sub-sampled {len(final_unlabeled_paths)} unlabeled files out of {len(cleaned_unlabeled)} possible."
    )
    return final_unlabeled_paths


def initialize_dataset(
    args,
    labeled_csv_paths_train,
    unlabeled_csv_paths_train,
    labeled_csv_paths_val,
    mask_tokens_dict,
):
    if args.model_choice == "any2any":
        if not all(x < y for x, y in zip(args.task_selection, args.task_selection[1:])):
            raise ValueError(
                "task_selection is not in increasing order, which is not compatible with __getitem__()"
            )
        if args.use_mav_for_emg == 1:
            # compute the effective length
            # scale down lambda, transition_buffer, etc.
            effective_mav_length = (
                args.window_size - args.inner_window_size
            ) // args.mav_inner_stride + 1
            args.lambda_poisson = 2
            scale_factor = effective_mav_length / args.window_size
            args.transition_buffer = max(1, int(args.transition_buffer * scale_factor))

        dataset_train = Any2Any_Dataset(
            labeled_csv_paths_train,
            unlabeled_csv_paths_train,
            args.median_filter_size,
            args.window_size,
            args.offset,
            args.embedding_method,
            args.lambda_poisson,
            False,
            args.sampling_probability_poisson,
            args.poisson_mask_percentage_sampling_range,
            args.end_mask_percentage_sampling_range,
            args.task_selection,
            args.stage_1_weights,
            args.stage_2_weights,
            args.mask_alignment,
            args.transition_buffer,
            mask_tokens_dict,
            args.with_training_curriculum,
            args.num_classes,
            args.medfilt_order,
            args.noise,
            args.hand_choice,
            args.inner_window_size,
            args.use_mav_for_emg,
        )
        # Note that for the validation set, we always fix the masks by setting seeded_mask = True
        # seeded_mask controls not just whether the masks are reproducible, but also if training uses a different mask for each sample at every epoch
        # If the mask is not seeded, then it's free to change over the epoch
        # Also, the unlabeled dataset have placeholder values
        dataset_val = Any2Any_Dataset(
            labeled_csv_paths_val,
            [],
            args.median_filter_size,
            args.window_size,
            args.offset,
            args.embedding_method,
            args.lambda_poisson,
            True,
            args.sampling_probability_poisson,
            args.poisson_mask_percentage_sampling_range,
            args.end_mask_percentage_sampling_range,
            args.task_selection,
            args.stage_1_weights,
            args.stage_2_weights,
            args.mask_alignment,
            args.transition_buffer,
            mask_tokens_dict,
            args.with_training_curriculum,
            args.num_classes,
            args.medfilt_order,
            0.0,  # no noise
            args.hand_choice,
            args.inner_window_size,
            args.use_mav_for_emg,
        )

    elif args.model_choice == "ed_tcn":
        dataset_train = EDTCN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_train,
            inner_window_size=150,
            inner_stride=25,
        )
        dataset_val = EDTCN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_val,
            inner_window_size=150,
            inner_stride=25,
        )
    elif args.model_choice == "lstm":
        dataset_train = LSTM_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            csv_paths=labeled_csv_paths_train,
            num_classes=args.num_classes,
            precomputed_mean=None,
            precomputed_std=None,
        )
        dataset_val = LSTM_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            csv_paths=labeled_csv_paths_val,
            num_classes=args.num_classes,
            precomputed_mean=dataset_train.global_mean,
            precomputed_std=dataset_train.global_std,
        )
    elif args.model_choice == "ann":
        dataset_train = ANN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_train,
            num_classes=args.num_classes,
            use_precomputed_stats=False,
        )
        dataset_val = ANN_Dataset(
            window_size=args.window_size,
            offset=args.offset,
            file_paths=labeled_csv_paths_val,
            num_classes=args.num_classes,
            use_precomputed_stats=True,
            precomputed_mean=dataset_train.mean_,
            precomputed_std=dataset_train.std_,
        )
    else:
        raise ValueError(f"Unknown model_choice: {args.model_choice}")

    return dataset_train, dataset_val


def initialize_model(args):
    if args.model_choice == "any2any":
        model = Any2Any_Model(
            args.embedding_dim,
            args.nhead,
            args.dropout,
            args.activation,
            args.num_layers,
            args.window_size,
            args.embedding_method,
            args.mask_alignment,
            args.share_pe,
            args.tie_weight,
            args.use_decoder,
            args.use_input_layernorm,
            args.num_classes,
            args.output_reduction_method,
            args.chunk_size,
            args.inner_window_size,
            args.use_mav_for_emg,
            args.mav_inner_stride,
        )
    elif args.model_choice == "ed_tcn":
        model = EDTCN_Model(
            num_channels=8,
            num_classes=args.num_classes,
            enc_filters=(128, 288),
            kernel_size=9,
        )
    elif args.model_choice == "lstm":
        model = LSTM_Model(
            input_size=8,
            fc_size=400,
            hidden_size=256,
            num_classes=args.num_classes,
        )
    elif args.model_choice == "ann":
        model = ANN_Model(num_classes=args.num_classes)
    else:
        raise ValueError(f"Unknown model_choice: {args.model_choice}")

    return model


def lr_lambda_cosine(current_step, warmup_steps, total_steps):
    """
    Cosine schedule with warmup:
    During warmup_steps: increases linearly from 0 to 1
    After warmup_steps:  0.5*(1 + cos(pi * progress))
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # Cosine decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_lambda_linear(current_step, warmup_steps, total_steps):
    """
    Linear schedule with warmup:
    During warmup_steps: increases linearly from 0 to 1
    After warmup_steps:  decreases linearly from 1 to 0
    """
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # Linear decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    return max(0.0, 1.0 - progress)


def lr_lambda_exponential(current_step, warmup_steps, total_steps, decay_rate):
    """
    Exponential schedule with linear warmup:
      - During warmup_steps: increases LR linearly from 0 to 1.
      - After warmup_steps:  decays LR via exp(-decay_rate * progress),
                             where progress goes from 0 to 1 across the
                             remaining steps.
    """
    if current_step < warmup_steps:
        # Linear warmup
        return float(current_step) / float(max(1, warmup_steps))
    # Exponential decay
    progress = float(current_step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    # progress goes from 0 at step == warmup_steps to 1 at step == total_steps
    return math.exp(-decay_rate * progress)
