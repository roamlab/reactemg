import argparse
import numpy as np
import random
import os
import torch.optim as optim
import sys
import torch
import torch.nn as nn
from preprocessing_utils import (
    parse_tuple,
    get_csv_paths,
    get_unlabeled_csv_paths,
    initialize_dataset,
    initialize_model,
    lr_lambda_cosine,
    lr_lambda_linear,
    lr_lambda_exponential,
)
from torch.optim.lr_scheduler import LambdaLR
from train_eval import initialize_training
from minlora import add_lora, get_lora_params, LoRAParametrization
from functools import partial


def main(args):
    # GPU init and anomaly detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)
    torch.autograd.set_detect_anomaly(True)

    # Seeding
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Record commandline input for logging
    command_line_command = "python3 " + " ".join(sys.argv)

    # Collect data paths
    labeled_csv_paths_train = []
    labeled_csv_paths_val = []
    unlabeled_csv_paths_train = []
    roam_data_master_folder = "../data/ROAM_EMG"
    roam_data_subfolders = [f"s{i}" for i in range(1, 29)]
    public_data_folders = [
        "../data/SS-STM_for_Myo_filtered",
        "../data/Mangalore_University",
        "../data/ROSHAMBO",
    ]
    epn_data_master_folder = "../data/EMG-EPN-612"
    labeled_csv_paths_train, labeled_csv_paths_val = get_csv_paths(
        dataset_selection=args.dataset_selection,
        num_classes=args.num_classes,
        roam_data_master_folder=roam_data_master_folder,
        roam_data_subfolders=roam_data_subfolders,
        public_data_folders=public_data_folders,
        epn_data_master_folder=epn_data_master_folder,
        val_patient_ids=args.val_patient_ids,
        epn_subset_percentage=args.epn_subset_percentage,
        discard_labeled_percentage=args.discard_labeled_percentage,
        custom_data_folder=args.custom_data_folder,
    )

    # Filter out unlabeled from both train/val, specific to the EMG-EPN-612 dataset
    # Not to be confused with not using unlabeled data. In our custom pipeline, we append "unlabel" to all EMG-EPN-612 files that have held-out labels
    labeled_csv_paths_train = [
        p
        for p in labeled_csv_paths_train
        if "unlabel" not in os.path.basename(p).lower()
    ]
    labeled_csv_paths_val = [
        p for p in labeled_csv_paths_val if "unlabel" not in os.path.basename(p).lower()
    ]
    print(f"labeled CSVs in training set: {len(labeled_csv_paths_train)}")
    print(f"labeled CSVs in validation set: {len(labeled_csv_paths_val)}")

    # Unlabeled processing
    if args.model_choice == "any2any" and 3 in args.task_selection:
        unlabeled_csv_paths_train = get_unlabeled_csv_paths(
            unlabeled_data_folder="../data/unlabeled_data",
            labeled_paths_train=labeled_csv_paths_train,
            labeled_paths_val=labeled_csv_paths_val,
            epn_unlabeled_classes=args.epn_unlabeled_classes,
            unlabeled_percentage=args.unlabeled_percentage,
        )
        unlabeled_csv_paths_train = sorted(list(set(unlabeled_csv_paths_train)))
        print(f"Final total unlabeled files: {len(unlabeled_csv_paths_train)}")

    # Define mask tokens
    if args.num_classes not in [3, 6]:
        raise Exception("num_classes not 3 nor 6")
    mask_tokens_dict = {
        "linear_projection": {
            "Action_mask": args.num_classes,
            "EMG_mask": 0,
        }
    }

    # Create torch dataset class
    print("Constructing Datasets:")
    dataset_train, dataset_val = initialize_dataset(
        args,
        labeled_csv_paths_train,
        unlabeled_csv_paths_train,
        labeled_csv_paths_val,
        mask_tokens_dict,
    )
    print(f"Total number of samples in training: {len(dataset_train)}")
    print(f"Total number of samples in validation: {len(dataset_val)}")
    print(f"Number of unlabeled samples in training: {len(unlabeled_csv_paths_train)}")

    # Model init
    model = initialize_model(args)

    # Loading local checkpoint if provided
    if args.saved_checkpoint_pth is not None:
        print(
            "saved_checkpoint_pth is not None. Loading pretrained weights from local..."
        )
        load_checkpoint_path = args.saved_checkpoint_pth
        checkpoint = torch.load(
            load_checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(
            checkpoint["model_info"]["model_state_dict"], strict=False
        )
    else:
        print("saved_checkpoint_pth is None. Initializing weights from scratch...")
    
    # Initialize LoRA
    if args.use_lora == 1:
        lora_config = {
            nn.Linear: {
                "weight": partial(
                    LoRAParametrization.from_linear,
                    rank=args.lora_rank,
                    lora_alpha=args.lora_alpha,
                    lora_dropout_p=args.lora_dropout_p,
                ),
            },
        }
        add_lora(model, lora_config)

    model.to(device)
    print(
        "Total trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # Optimizer Initialization
    # LoRA compatible, with warmup and decay
    total_steps = args.epochs * (len(dataset_train) // args.batch_size)
    warmup_steps = int(args.warmup_ratio * total_steps)
    if args.use_lora == 1:
        print("LORA Activated")
        lora_parameters = [
            {"params": list(get_lora_params(model))},
        ]
        optimizer = optim.AdamW(lora_parameters, lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    if args.use_warmup_and_decay:
        if args.lr_decay_method == "cosine":
            scheduler = LambdaLR(
                optimizer,
                lambda step: lr_lambda_cosine(step, warmup_steps, total_steps),
            )
        elif args.lr_decay_method == "linear":
            scheduler = LambdaLR(
                optimizer,
                lambda step: lr_lambda_linear(step, warmup_steps, total_steps),
            )
        elif args.lr_decay_method == "exponential":
            scheduler = LambdaLR(
                optimizer,
                partial(
                    lr_lambda_exponential,
                    warmup_steps,
                    total_steps,
                    args.exponential_decay_rate,
                ),
            )
        else:
            raise ValueError("lr_decay_method is not recognized")
    else:
        scheduler = None

    # Constructing args dictionary for saving to local
    args_dict = vars(args)
    args_dict["command_line_command"] = command_line_command
    args_dict["mask_tokens_dict"] = mask_tokens_dict
    args_dict["labeled_csv_paths_train"] = (labeled_csv_paths_train,)
    args_dict["labeled_csv_paths_val"] = (labeled_csv_paths_val,)
    args_dict["unlabeled_csv_paths_train"] = (unlabeled_csv_paths_train,)
    args_dict["public_data_folders"] = public_data_folders
    args_dict["num_files_train"] = len(labeled_csv_paths_train)
    args_dict["num_files_val"] = len(labeled_csv_paths_val)
    args_dict["num_samples_train"] = len(dataset_train)
    args_dict["num_samples_val"] = len(dataset_val)
    if args.model_choice == "lstm":
        args_dict["precomputed_mean"] = dataset_train.global_mean
        args_dict["precomputed_std"] = dataset_train.global_std
    if args.model_choice == "ann":
        args_dict["precomputed_mean"] = dataset_train.mean_
        args_dict["precomputed_std"] = dataset_train.std_

    # Training
    initialize_training(
        args.model_choice,
        model,
        dataset_train,
        dataset_val,
        optimizer,
        scheduler,
        args.epochs,
        device,
        args_dict,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #########################################
    ############ Global Settings ############
    #########################################
    parser.add_argument("--seed", default=42, type=int, help="Seed to use")
    parser.add_argument(
        "--saved_checkpoint_pth",
        type=str,
        help="If not None, load pretrained weights that are saved on local. Otherwise, train from scratch",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        type=str,
        help="Experiment name that is used to save to wandb. If not provided, auto-generate",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of intent classes to perform classifications upon. Can be 3 or 6",
    )
    parser.add_argument(
        "--model_choice",
        type=str,
        required=True,
        help="Which model architecture to use",
    )

    ##################################################
    ############ Preprocessing Parameters ############
    ##################################################
    parser.add_argument(
        "--window_size",
        type=int,
        required=True,
        help="Number of timesteps to use for one sample",
    )
    parser.add_argument(
        "--median_filter_size", default=3, type=int, help="Width of the median filter"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=30,
        help="Step to skip before extracting the next sample",
    )
    parser.add_argument(
        "--use_wrist",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether or not to use wrist movement data. 0 don't include them.",
    )
    parser.add_argument(
        "--transition_buffer",
        default=50,
        type=int,
        help="If a transition sample fall within the first transition_buffer timesteps or the last transition_buffer timesteps, treat it as a non-transition sample",
    )
    parser.add_argument(
        "--medfilt_order",
        default="before_rec",
        type=str,
        choices=["before_rec", "after_rec"],
        help="Decide if doing median filter before or after rectification",
    )
    parser.add_argument(
        "--dataset_selection",
        required=True,
        type=str,
        choices=[
            "roam_only",
            "pub_only",
            "epn_only",
            "pub_with_roam",
            "pub_with_roam_with_epn",
            "pub_with_epn",
            "custom_folder",
        ],
        help="Select the dataset used to train the current model",
    )
    parser.add_argument(
        "--custom_data_folder",
        type=str,
        default=None,
        help="Path to custom folder containing .csv or .npy files. Required when dataset_selection='custom_folder'",
    )
    parser.add_argument(
        "--val_patient_ids",
        nargs="+",
        required=True,
        help="List of two-letter patient IDs to include in validation",
    )
    parser.add_argument(
        "--epn_subset_percentage",
        default=0.02,
        type=float,
        help="Percentage of the total EPN samples to use, when training together with pub and roam",
    )
    parser.add_argument(
        "--hand_choice",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Method used for embedding the time series",
    )
    parser.add_argument(
        "--discard_labeled_percentage",
        type=float,
        default=0.0,
        help="Fraction of labeled training data to discard at the file level (0.0 to 1.0).",
    )
    parser.add_argument(
        "--epn_unlabeled_classes",
        type=int,
        choices=[3, 6],
        default=3,
        help="Specifies how many gestures to keep for EPN unlabeled. 3 => skip wave/pinch, 6 => keep them.",
    )
    parser.add_argument(
        "--unlabeled_percentage",
        type=float,
        default=0.0,
        help="Fraction of unlabeled data to use (0.0 means no unlabeled data being used).",
    )

    ##########################################
    ############ Model Parameters ############
    ##########################################
    parser.add_argument(
        "--embedding_method",
        type=str,
        default="linear_projection",
        help="Method used for embedding the time series",
    )
    parser.add_argument(
        "--embedding_dim",
        default=128,
        type=int,
        help="Size of the hidden states in the embedding layers, also used for transformer",
    )
    parser.add_argument(
        "--nhead", default=4, type=int, help="Number of multiheaded attention"
    )
    parser.add_argument(
        "--dropout", default=0.15, type=float, help="Dropout for transformer layers"
    )
    parser.add_argument(
        "--activation",
        default="gelu",
        type=str,
        help="Activation function for transformer layers",
    )
    parser.add_argument(
        "--num_layers",
        default=2,
        type=int,
        help="Number of layers for transformer encoder and decoder",
    )
    parser.add_argument(
        "--share_pe",
        action="store_true",
        help="Setting to True shares the positional encoding layer between EMG and action",
    )
    parser.add_argument(
        "--tie_weight",
        action="store_true",
        help="Setting to True ties the weight of action embedding with action out-projection. If using separate_channel as embedding method, also ties EMG's input embedding and output projection",
    )
    parser.add_argument(
        "--use_decoder",
        action="store_true",
        help="Setting to True creates two decoder blocks that takes EMG embedding and action embedding to generate",
    )
    parser.add_argument(
        "--use_lora",
        default=0,
        type=int,
        choices=[0, 1],
        help="Whether or not to use lora",
    )
    parser.add_argument("--lora_rank", default=16, type=int, help="lora rank r")
    parser.add_argument("--lora_alpha", default=8, type=int, help="lora alpha")
    parser.add_argument(
        "--lora_dropout_p", default=0.05, type=float, help="dropout rate on lora"
    )
    parser.add_argument(
        "--output_reduction_method",
        default="none",
        type=str,
        choices=["none", "learned", "pooling"],
        help="method to reduce output prediction size",
    )
    parser.add_argument(
        "--chunk_size",
        default=0,
        type=int,
        help="how many steps the output reduction account for",
    )

    ##############################################
    ############# Training Parameters ############
    ##############################################
    parser.add_argument("--batch_size", default=128, type=int, help="Batch Size")
    parser.add_argument("--epochs", default=12, type=int, help="Training epochs")
    parser.add_argument(
        "--learning_rate", default=1e-4, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--use_input_layernorm",
        action="store_true",
        help="Setting to True adds an input layernorm",
    )
    parser.add_argument(
        "--use_warmup_and_decay",
        action="store_true",
        help="Setting to True uses learning rate warmup and decay",
    )
    parser.add_argument(
        "--warmup_ratio",
        default=0.05,
        type=float,
        help="proportion of training steps to use as warmup. step means a per-batch basis",
    )
    parser.add_argument(
        "--with_training_curriculum",
        action="store_true",
        help="Setting to True applies training curriculum",
    )
    parser.add_argument(
        "--use_classifier_loss",
        action="store_true",
        help="Setting to True applies an additional auxiliary loss term that functions as a classifier loss on sequence level",
    )
    parser.add_argument(
        "--classifier_loss_weight",
        default=1,
        type=float,
        help="Weighting applied to the classifier loss",
    )
    parser.add_argument(
        "--split_unlabeled_batch",
        action="store_true",
        help="Setting to True makes every batch contain either fully labeled or fully unlabeled data. Leaving as False means each batch can potentially contain both labeled and unlabeled data",
    )
    parser.add_argument(
        "--cost_sensitive_loss",
        action="store_true",
        help="Setting to True uses the cost matrix to compute a cost-sensitive loss",
    )
    parser.add_argument(
        "--unmask_transition_range",
        default=0,
        type=int,
        help="Setting to a non-zero value unmasks unmask_transition_range timesteps around a transition, meaning that transition timesteps are never backpropagated, which effectively functions as a stop-gradient",
    )
    parser.add_argument(
        "--label_smoothing_range",
        default=0,
        type=int,
        help="Setting to a non-zero value smoothes label_smoothing_range after a transition and label_smoothing_range // 2 range before the transition. Loss computed via KL divergence",
    )
    parser.add_argument(
        "--lambda_poisson", default=7, type=float, help="Average masked span length"
    )
    parser.add_argument(
        "--sampling_probability_poisson",
        default=0.99,
        type=float,
        help="Probability of selecting the Poisson Lambda mask type. The remaining probability is the probability of selecting end masking as the type",
    )
    parser.add_argument(
        "--poisson_mask_percentage_sampling_range",
        nargs="+",
        default=[(0.3, 0.75), (0.3, 0.75), (0.3, 0.75), (0.3, 0.75), (0.3, 0.75)],
        type=parse_tuple,
        help="List of tuples in the form of (lower, upper), which controls the range of mask percentages for poisson lambda masking to sample from",
    )
    parser.add_argument(
        "--end_mask_percentage_sampling_range",
        nargs="+",
        default=[(0.3, 0.6), (0.3, 0.6), (0.3, 0.6), (0.3, 0.6), (0.3, 0.6)],
        type=parse_tuple,
        help="List of tuples in the form of (lower, upper), which controls the range of mask percentages to end masking sample from",
    )
    parser.add_argument(
        "--task_selection",
        nargs="+",
        type=int,
        help="a list of integers specifying the subset of masking strategies that will be applied to the training data. [0,1,2,3,4] means everything. For int-task matches, check the slide deck",
    )
    parser.add_argument(
        "--stage_1_weights",
        nargs="+",
        type=float,
        default=[0.5, 0.5],
        help="a list of floats indicating the weights use to sample from the curriculum in stage 1",
    )
    parser.add_argument(
        "--stage_2_weights",
        nargs="+",
        type=float,
        default=[0.01, 0.99],
        help="a list of floats indicating the weights use to sample from the curriculum in stage 2",
    )
    parser.add_argument(
        "--mask_alignment",
        default="aligned",
        type=str,
        help="Whether the masks are temporally 'aligned' or 'non-aligned'",
    )
    parser.add_argument(
        "--noise",
        default=0.0,
        type=float,
        help="Uniform noise amplitude. 0.0 means no noise; e.g., 0.01 means [-0.01, +0.01].",
    )
    parser.add_argument(
        "--lr_decay_method",
        default="linear",
        type=str,
        choices=["cosine", "linear", "exponential"],
        help="LR decay method. Warmup is always linear",
    )
    parser.add_argument(
        "--exponential_decay_rate",
        default=2.0,
        type=float,
        help="Decay factor for exponential LR schedule.",
    )
    parser.add_argument(
        "--scale_emg_loss",
        default=0,
        type=int,
        choices=[0, 1],
        help="If 1, scale emg loss (MSE) by 100 times to match the scale of the CE loss on actions",
    )
    parser.add_argument(
        "--inner_window_size",
        type=int,
        required=True,
        help="If equal to window_size, do old pipeline. Otherwise, must be a divisor of window_size, and less than window_size.",
    )
    parser.add_argument(
        "--use_mav_for_emg",
        type=int,
        default=0,
        choices=[0, 1],
        help="If 1, extract MAV subwindows (ED-TCN style) for the EMG sequence instead of raw timesteps.",
    )
    parser.add_argument(
        "--mav_inner_stride",
        type=int,
        default=25,
        help="Stride for MAV subwindow when use_mav_for_emg=1, analogous to ED-TCN's inner_stride (e.g. 25).",
    )

    args = parser.parse_args()

    # Custom folder safety check
    if args.dataset_selection == "custom_folder" and args.custom_data_folder is None:
        parser.error("--dataset_selection is custom_folder, but no --custom_data_folder is provided.")

    # LoRA safety check
    if args.use_lora == 1 and args.saved_checkpoint_pth is None:
        raise Exception(
            "Attempting to use lora without providing a pretrained checkpoint"
        )

    # Model safety check
    if args.model_choice == "any2any" and args.task_selection is None:
        parser.error("--model_choice is any2any, but no --task_selection is provided.")
    if args.inner_window_size == args.window_size:
        print("inner_window_size == window_size: Using main pipeline...")
    elif args.inner_window_size > args.window_size:
        raise ValueError(
            f"inner_window_size ({args.inner_window_size}) cannot exceed window_size ({args.window_size})"
        )
    elif (args.window_size % args.inner_window_size) != 0:
        raise ValueError(
            f"window_size ({args.window_size}) is not divisible by inner_window_size ({args.inner_window_size})"
        )
    else:
        print("inner_window_size < window_size: Activating coarse-resolution method.")

    main(args)
