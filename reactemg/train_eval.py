import os, socket, datetime, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Sampler
import wandb
from minlora import get_lora_state_dict, name_is_lora


def initialize_training(
    model_choice,
    model,
    dataset_train,
    dataset_val,
    optimizer,
    scheduler,
    epochs,
    device,
    args_dict,
):
    if model_choice == "any2any":
        train_any2any(
            model,
            dataset_train,
            dataset_val,
            optimizer,
            scheduler,
            epochs,
            device,
            args_dict,
        )
    elif model_choice == "ed_tcn":
        train_edtcn(
            model,
            dataset_train,
            dataset_val,
            optimizer,
            scheduler,
            epochs,
            device,
            args_dict,
        )
    elif model_choice == "lstm":
        train_lstm(
            model,
            dataset_train,
            dataset_val,
            optimizer,
            scheduler,
            epochs,
            device,
            args_dict,
        )
    elif model_choice == "ann":
        train_ann(
            model,
            dataset_train,
            dataset_val,
            optimizer,
            scheduler,
            epochs,
            device,
            args_dict,
        )
    elif model_choice == "trahgr":
        train_trahgr(
            model,
            dataset_train,
            dataset_val,
            optimizer,
            scheduler,
            epochs,
            device,
            args_dict,
        )
    else:
        raise ValueError(f"Unknown model_choice: {model_choice}")

    return model


################################################
############### Any2Any Training ###############
################################################
class LabeledUnlabeledSampler(Sampler):
    def __init__(self, labeled_indices, unlabeled_indices, batch_size):
        self.labeled_indices = labeled_indices
        self.unlabeled_indices = unlabeled_indices
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.labeled_indices)
        random.shuffle(self.unlabeled_indices)

        # Create labeled and unlabeled batches
        labeled_batches = [
            self.labeled_indices[i : i + self.batch_size]
            for i in range(0, len(self.labeled_indices), self.batch_size)
        ]
        unlabeled_batches = [
            self.unlabeled_indices[i : i + self.batch_size]
            for i in range(0, len(self.unlabeled_indices), self.batch_size)
        ]

        # Combine and shuffle batches
        all_batches = labeled_batches + unlabeled_batches
        random.shuffle(all_batches)

        # Flatten the list of batches
        return iter(all_batches)

    def __len__(self):
        return (
            len(self.labeled_indices) + len(self.unlabeled_indices)
        ) // self.batch_size


def compute_parameter_changes(model, initial_lora_params, initial_non_lora_params):
    lora_changes = {}
    non_lora_changes = {}

    for name, param in model.named_parameters():
        if name_is_lora(name):
            # Compute change for LoRA parameters
            change = torch.norm(param - initial_lora_params[name]).item()
            lora_changes[name] = change
        else:
            # Compute change for non-LoRA parameters
            change = torch.norm(param - initial_non_lora_params[name]).item()
            non_lora_changes[name] = change

    return lora_changes, non_lora_changes


def process_batch(batch_data, model, args_dict, device, cost_sensitive_loss):
    """
    A unified function for handling both:
      - Fine-resolution (inner_window_size == window_size), which the dataset returns as 8 items:
          (emg_window, action_window, masked_emg, masked_actions,
           mask_positions_emg, mask_positions_actions, task_idx, transition_index)

        'masked_emg' and 'masked_actions' were numeric masked versions
        from the dataset. But we don't feed them into the model a second time,
        or else we'd be double-masking the EMG. We'll feed raw data to the model,
        and then rely on `mask_positions_emg` to tell the model which timesteps to
        replace with a learned mask token.

      - Coarse-resolution (inner_window_size < window_size), which the dataset returns as 7 items:
          (emg_window, coarse_action_window, masked_coarse_actions,
           mask_positions_coarse_emg, mask_positions_coarse_actions, task_idx, transition_index)

        Here, numeric masking for action tokens may happen in the dataset (i.e. masked_coarse_actions),
        but we only do the "numeric mask" for EMG inside the model (downsample + replace).
        Then we do the standard expansions to compute MSE/CE at the correct times.

    It returns:
       emg_loss, action_loss, total_loss, mask_positions_emg, mask_positions_actions
       so the training code can track them or compute how many positions got masked, etc.
    """

    # Check the length of batch_data to see which branch we are in
    if len(batch_data) == 8:
        # =============== FINE-RESOLUTION BRANCH ===============
        (
            emg_window,  # (B, window_size, 8) -> raw
            action_window,  # (B, window_size)
            masked_emg_dataset,  # (B, window_size, 8) -> numeric masked from dataset
            masked_actions_dataset,  # (B, window_size) -> numeric masked from dataset
            mask_positions_emg,  # (B, window_size, 8) bool
            mask_positions_actions,  # (B, window_size) bool
            task_idx,
            transition_index,
        ) = batch_data

        emg_window = emg_window.to(device)
        action_window = action_window.to(device)
        masked_emg_dataset = masked_emg_dataset.to(device)
        masked_actions_dataset = masked_actions_dataset.to(device)

        mask_positions_emg = mask_positions_emg.to(device)
        mask_positions_actions = mask_positions_actions.to(device)
        task_idx = task_idx.to(device)
        transition_index = transition_index.to(device)

        # 2) Model forward
        # For the fine-resolution (per timestep branch), do we masking inside the model.
        # Therefore, we pass raw emg_window and raw action_window
        # The boolean mask (mask_positions_emg, mask_positions_actions)
        # will tell the model which positions to replace with the learned mask token.
        emg_output, action_output = model(
            emg_window,
            masked_actions_dataset,
            task_idx,
            mask_positions_emg,
            return_output=False,
            emg_window=emg_window,
            action_window=action_window,
        )
        # Shapes typically:
        #   emg_output -> (B, window_size, 8)
        #   action_output -> (B, window_size, vocab_size), unless chunking or pooling

        # 3) Compute EMG Loss by flattening & indexing masked positions
        #    We want MSE only on the timesteps that were masked,
        #    i.e. mask_positions_emg == True.
        B, W, C = emg_output.shape  # (B, window_size, 8)
        emg_output_flat = emg_output.reshape(B * W, C)  # (B*W, 8)
        emg_window_flat = emg_window.reshape(B * W, C)  # (B*W, 8)
        mask_positions_e = mask_positions_emg.reshape(B * W, C)  # bool, same shape

        # Gather
        emg_output_masked = emg_output_flat[mask_positions_e]
        emg_gt_masked = emg_window_flat[mask_positions_e]

        if emg_output_masked.numel() > 0:
            emg_loss = F.mse_loss(emg_output_masked, emg_gt_masked)
        else:
            emg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 4) Compute Action Loss
        #    Flatten (B, W, vocab_size) => (B*W, vocab_size).
        #    Then gather only masked positions where mask_positions_actions == True.
        B, W, vocab_size = action_output.shape
        action_output_flat = action_output.reshape(B * W, vocab_size)
        action_window_flat = action_window.reshape(B * W)
        mask_positions_a = mask_positions_actions.reshape(B * W)  # bool

        action_output_masked = action_output_flat[mask_positions_a]
        action_gt_masked = action_window_flat[mask_positions_a]

        if action_output_masked.numel() > 0:
            if args_dict["cost_sensitive_loss"]:
                action_loss = cost_sensitive_loss(
                    action_output_masked, action_gt_masked
                )
            else:
                action_loss = F.cross_entropy(action_output_masked, action_gt_masked)
        else:
            action_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 5) Combine
        if args_dict["scale_emg_loss"] == 1:
            total_loss = 100.0 * emg_loss + action_loss
        else:
            total_loss = emg_loss + action_loss

        return (
            emg_loss,
            action_loss,
            total_loss,
            mask_positions_emg,
            mask_positions_actions,
        )

    else:
        # =============== COARSE-RESOLUTION BRANCH ===============
        (
            emg_window,  # (B, window_size, 8) raw
            coarse_action_window,  # (B, coarse_len)
            masked_coarse_actions,  # (B, coarse_len), possibly numeric masked for actions
            mask_positions_coarse_emg,  # (B, coarse_len)
            mask_positions_coarse_actions,  # (B, coarse_len)
            task_idx,
            transition_index,
        ) = batch_data

        # 1) Move to device
        emg_window = emg_window.to(device)
        coarse_action_window = coarse_action_window.to(device)
        masked_coarse_actions = masked_coarse_actions.to(device)
        mask_positions_coarse_emg = mask_positions_coarse_emg.to(device)
        mask_positions_coarse_actions = mask_positions_coarse_actions.to(device)
        task_idx = task_idx.to(device)
        transition_index = transition_index.to(device)

        # 2) Model forward
        #    In coarse mode, the model does the strided conv downsample on emg_window,
        #    then replaces entire sub-windows with a learned mask if mask_positions_coarse_emg == True,
        #    then upsamples back to (B, window_size, 8).
        #    Meanwhile, the action branch uses masked_coarse_actions if the dataset masked them.
        emg_output, action_output = model(
            emg_window,
            masked_coarse_actions,
            task_idx,
            mask_positions_coarse_emg,
            return_output=False,
            emg_window=None,
            action_window=None,
        )
        # => emg_output: (B, window_size, 8)
        # => action_output: (B, coarse_len, vocab_size)

        # 3) Compute Action Loss at coarse scale
        B, coarse_len, vocab_size = action_output.shape
        action_output_flat = action_output.reshape(B * coarse_len, vocab_size)

        # Flatten GT
        coarse_action_gt_flat = coarse_action_window.reshape(B * coarse_len)

        # Gather only where mask_positions_coarse_actions is True
        mask_positions_coarse_actions = mask_positions_coarse_actions.reshape(
            B * coarse_len
        )
        action_output_masked = action_output_flat[mask_positions_coarse_actions]
        action_gt_masked = coarse_action_gt_flat[mask_positions_coarse_actions]

        if action_output_masked.numel() > 0:
            if args_dict["cost_sensitive_loss"]:
                action_loss = cost_sensitive_loss(
                    action_output_masked, action_gt_masked
                )
            else:
                action_loss = F.cross_entropy(action_output_masked, action_gt_masked)
        else:
            action_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 4) Compute EMG Loss in full resolution
        #    We only do MSE for sub-windows that were masked. Expand the mask:
        #    i.e. for each coarse index i, [i*inner_ws : (i+1)*inner_ws] => True if coarse i is masked.
        B, full_w, c = emg_output.shape  # (B, window_size, 8)
        raw_mask = torch.zeros((B, full_w), dtype=torch.bool, device=device)

        inner_ws = args_dict["inner_window_size"]
        for b in range(B):
            masked_indices = torch.where(mask_positions_coarse_emg[b])[
                0
            ]  # e.g. sub-window indices
            for i_coarse in masked_indices:
                start_t = i_coarse.item() * inner_ws
                end_t = start_t + inner_ws
                raw_mask[b, start_t:end_t] = True

        # Flatten
        emg_output_flat = emg_output.reshape(B * full_w, c)
        emg_gt_flat = emg_window.reshape(B * full_w, c)

        emg_output_masked = emg_output_flat[raw_mask.reshape(-1)]
        emg_gt_masked = emg_gt_flat[raw_mask.reshape(-1)]

        if emg_output_masked.numel() > 0:
            emg_loss = F.mse_loss(emg_output_masked, emg_gt_masked)
        else:
            emg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 5) Combine
        if args_dict["scale_emg_loss"] == 1:
            total_loss = 100.0 * emg_loss + action_loss
        else:
            total_loss = emg_loss + action_loss

        return (
            emg_loss,
            action_loss,
            total_loss,
            mask_positions_coarse_emg,
            mask_positions_coarse_actions,
        )


def train_one_epoch(
    model,
    dataloader_train,
    optimizer,
    scheduler,
    device,
    args_dict,
    cost_sensitive_loss,
    global_step,
):
    """
    Runs one epoch of training.
    - process_batch() returns (emg_loss, action_loss, total_loss, mask_positions_emg, mask_positions_actions).
    - We multiply by the number of masked positions so that when we compute the average at the end,
      we get a proper weighting by how many items were masked.
    """

    model.train()

    total_loss_acc = 0.0
    total_emg_loss_acc = 0.0
    total_action_loss_acc = 0.0
    total_masked_positions = 0

    grad_norms = []

    for batch_data in dataloader_train:
        optimizer.zero_grad()

        emg_loss, action_loss, combined_loss, mask_emg, mask_action = process_batch(
            batch_data, model, args_dict, device, cost_sensitive_loss
        )

        combined_loss.backward()

        # Compute gradient norm
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm**0.5
        grad_norms.append(grad_norm)
        optimizer.step()

        # Step the scheduler each iteration if used
        if args_dict["use_warmup_and_decay"]:
            scheduler.step()
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            if global_step % 500 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Global Step {global_step}, Learning Rate: {current_lr}")

        # Count the number of masked positions in both EMG and actions
        # Note: shape might be (B, window_size, 8) in fine or (B, coarse_length) in coarse
        # sum() works fine in both cases:
        masked_count = mask_emg.sum().item() + mask_action.sum().item()

        # Accumulate weighted by masked_count
        total_loss_acc += combined_loss.item() * masked_count
        total_emg_loss_acc += emg_loss.item() * masked_count
        total_action_loss_acc += action_loss.item() * masked_count
        total_masked_positions += masked_count

    # Compute averaged losses
    if total_masked_positions > 0:
        avg_loss = total_loss_acc / total_masked_positions
        avg_emg_loss = total_emg_loss_acc / total_masked_positions
        avg_action_loss = total_action_loss_acc / total_masked_positions
    else:
        avg_loss = 0.0
        avg_emg_loss = 0.0
        avg_action_loss = 0.0

    # Compute gradient norm stats
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0.0
    max_grad_norm = max(grad_norms) if grad_norms else 0.0
    if args_dict["use_warmup_and_decay"]:
        lr_end_of_epoch = scheduler.get_last_lr()[0]
    else:
        lr_end_of_epoch = args_dict["learning_rate"]

    return {
        "total_loss": avg_loss,
        "emg_loss": avg_emg_loss,
        "action_loss": avg_action_loss,
        "masked_count": total_masked_positions,
        "avg_grad_norm": avg_grad_norm,
        "max_grad_norm": max_grad_norm,
        "global_step": global_step,
        "learning_rate": lr_end_of_epoch,
    }


def validate_one_epoch(
    model,
    dataloader_val,
    device,
    args_dict,
    cost_sensitive_loss,
):
    """
    Runs one epoch of validation (no gradient updates).
    """

    model.eval()

    total_loss_acc = 0.0
    total_emg_loss_acc = 0.0
    total_action_loss_acc = 0.0
    total_masked_positions = 0

    with torch.no_grad():
        for batch_data in dataloader_val:
            emg_loss, action_loss, combined_loss, mask_emg, mask_action = process_batch(
                batch_data, model, args_dict, device, cost_sensitive_loss
            )

            masked_count = mask_emg.sum().item() + mask_action.sum().item()
            total_loss_acc += combined_loss.item() * masked_count
            total_emg_loss_acc += emg_loss.item() * masked_count
            total_action_loss_acc += action_loss.item() * masked_count
            total_masked_positions += masked_count

    if total_masked_positions > 0:
        avg_loss = total_loss_acc / total_masked_positions
        avg_emg_loss = total_emg_loss_acc / total_masked_positions
        avg_action_loss = total_action_loss_acc / total_masked_positions
    else:
        avg_loss = 0.0
        avg_emg_loss = 0.0
        avg_action_loss = 0.0

    return {
        "total_loss": avg_loss,
        "emg_loss": avg_emg_loss,
        "action_loss": avg_action_loss,
        "masked_count": total_masked_positions,
    }


def train_any2any(
    model, dataset_train, dataset_val, optimizer, scheduler, epochs, device, args_dict
):
    # Initialize wandb
    machine_name = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["machine_name"] = machine_name
    if args_dict["exp_name"] is not None:
        exp_name = f"{args_dict['exp_name']}_{timestamp}_{machine_name}"
    else:
        exp_name = f"unnamed_{timestamp}_{machine_name}"
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "default_project"),
        entity="yolandaz",
        name=exp_name,
        config=args_dict,
    )

    # Initialize timestamped folder to save runs
    checkpoint_dir = f"model_checkpoints/{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args_dict["use_lora"] == 1:
        initial_lora_params = {}
        initial_non_lora_params = {}
        for name, param in model.named_parameters():
            if name_is_lora(name):
                initial_lora_params[name] = param.clone().detach()
            else:
                initial_non_lora_params[name] = param.clone().detach()
    # Marking where pretrained weights came from, if any
    if args_dict["saved_checkpoint_pth"] is not None:
        saved_weights_initialization_file_path = os.path.join(
            checkpoint_dir, "saved_weights_initialization_file_path.txt"
        )
        with open(saved_weights_initialization_file_path, "w") as file:
            file.write(args_dict["saved_checkpoint_pth"])

    # Global counter for every step
    global_step = 0

    # Defining cost-sensitive loss
    # override, not being used
    cost_sensitive_loss = None

    # Train and Validation Loop
    for epoch in range(epochs):
        # Apply updates to the dataset and construct new dataloaders for curriculum learning
        dataset_train.cur_epoch = epoch
        dataset_val.cur_epoch = epoch

        if args_dict["embedding_method"] == "linear_projection":
            if args_dict["with_training_curriculum"]:
                if epoch < epochs // 2:
                    dataset_train.curriculum_stage = 0
                    dataset_val.curriculum_stage = 0
                else:
                    dataset_train.curriculum_stage = 2
                    dataset_val.curriculum_stage = 2
            else:
                dataset_train.curriculum_stage = 2
                dataset_val.curriculum_stage = 2
        else:
            if args_dict["with_training_curriculum"]:
                if epoch < epochs // 3:
                    dataset_train.curriculum_stage = 0
                    dataset_val.curriculum_stage = 0
                elif epoch >= epochs // 3 and epoch < epochs // 3 * 2:
                    dataset_train.curriculum_stage = 1
                    dataset_val.curriculum_stage = 1
                else:
                    dataset_train.curriculum_stage = 2
                    dataset_val.curriculum_stage = 2
            else:
                dataset_train.curriculum_stage = 2
                dataset_val.curriculum_stage = 2

        # Create a dataloader at each iteration to account for updates in the dataset class
        if args_dict["split_unlabeled_batch"]:
            sampler = LabeledUnlabeledSampler(
                dataset_train.labeled_indices,
                dataset_train.unlabeled_indices,
                args_dict["batch_size"],
            )
            dataloader_train = DataLoader(
                dataset_train, batch_sampler=sampler, pin_memory=True, num_workers=4
            )
        else:
            dataloader_train = DataLoader(
                dataset_train,
                batch_size=args_dict["batch_size"],
                shuffle=True,
                pin_memory=True,
                num_workers=4,
            )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args_dict["batch_size"] // 2,
            shuffle=True,
            pin_memory=True,
            num_workers=4,
        )

        train_stats = train_one_epoch(
            model=model,
            dataloader_train=dataloader_train,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            args_dict=args_dict,
            cost_sensitive_loss=cost_sensitive_loss,
            global_step=global_step,
        )
        # Update global_step from returned dictionary
        global_step = train_stats["global_step"]

        # Periodically check parameter changes
        if args_dict["use_lora"] == 1:
            lora_changes, non_lora_changes = compute_parameter_changes(
                model, initial_lora_params, initial_non_lora_params
            )
            print(f"Epoch {epoch} - LoRA Parameter Changes:")
            for name, change in lora_changes.items():
                print(f"{name}: {change}")
            print()
            print(f"Epoch {epoch} - Non-LoRA Parameter Changes:")
            for name, change in non_lora_changes.items():
                print(f"{name}: {change}")
            print()

        val_stats = validate_one_epoch(
            model=model,
            dataloader_val=dataloader_val,
            device=device,
            args_dict=args_dict,
            cost_sensitive_loss=cost_sensitive_loss,
        )

        # Print / Log
        print(
            f"Epoch {epoch} -> Train Loss: {train_stats['total_loss']:.4f} | Val Loss: {val_stats['total_loss']:.4f}"
        )

        log_data = {
            "epoch": epoch,
            "training_loss": train_stats["total_loss"],
            "validation_loss": val_stats["total_loss"],
            "training_emg_loss": train_stats["emg_loss"],
            "training_action_loss": train_stats["action_loss"],
            "validation_emg_loss": val_stats["emg_loss"],
            "validation_action_loss": val_stats["action_loss"],
            "average_gradient_norm": train_stats["avg_grad_norm"],
            "max_gradient_norm": train_stats["max_grad_norm"],
            "saved_checkpoint_pth": args_dict["saved_checkpoint_pth"],
            "learning_rate": train_stats["learning_rate"],
        }
        wandb.log(log_data)

        # Save model checkpoint
        current_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        model_info = {
            "model_state_dict": model.state_dict(),
        }
        save_dict = {
            "model_info": model_info,
            "args_dict": args_dict,
        }
        if args_dict["use_lora"] == 1:
            save_dict["lora_state_dict"] = get_lora_state_dict(model)
        torch.save(save_dict, current_checkpoint_path)

    wandb.finish()


def train_edtcn(
    model, dataset_train, dataset_val, optimizer, scheduler, epochs, device, args_dict
):
    """
    A baseline training function for ED-TCN that:
      - Initializes a W&B run
      - Creates DataLoaders from dataset_train and dataset_val
      - Trains for "epochs" epochs
      - Logs metrics (losses, LR, grad norms) to wandb
      - Saves checkpoints
      - Supports an optional scheduler (may be None)
    """

    # Retrieve batch_size from args_dict or default to 100, as used in ED-TCN
    batch_size = args_dict.get("batch_size", 100)

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    machine_name = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["machine_name"] = machine_name

    if args_dict["exp_name"] is not None:
        exp_name = f"{args_dict['exp_name']}_{timestamp}_{machine_name}"
    else:
        exp_name = f"unnamed_{timestamp}_{machine_name}"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "default_project"),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=exp_name,
        config=args_dict,
    )

    checkpoint_dir = f"model_checkpoints/{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        grad_norms = []

        for i, (X, Y, raw_label_seq) in enumerate(train_loader):
            X = X.to(device)  # shape (B, T, 8)
            Y = Y.to(device)  # shape (B, T)

            logits = model(X)
            logits = logits.permute(0, 2, 1).contiguous()

            loss = criterion(logits, Y)

            optimizer.zero_grad()
            loss.backward()

            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.data.norm(2).item())

            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        avg_grad_norm = (sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0
        max_grad_norm = (max(grad_norms)) if grad_norms else 0.0

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for i, (Xv, Yv, raw_label_seq) in enumerate(val_loader):
                Xv = Xv.to(device)
                Yv = Yv.to(device)

                logits_v = model(Xv)  # shape (B, T, num_classes)
                logits_v = logits_v.permute(0, 2, 1).contiguous()
                loss_v = criterion(logits_v, Yv)
                running_val_loss += loss_v.item()
        avg_val_loss = running_val_loss / len(val_loader)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
                "average_grad_norm": avg_grad_norm,
                "max_grad_norm": max_grad_norm,
            }
        )

        print(
            f"[Epoch {epoch}/{epochs}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.6f} | "
            f"Avg Grad Norm: {avg_grad_norm:.4f}"
        )

        current_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        model_info = {
            "model_state_dict": model.state_dict(),
        }
        save_dict = {
            "model_info": model_info,
            "args_dict": args_dict,
        }
        torch.save(save_dict, current_checkpoint_path)

    wandb.finish()


def train_lstm(
    model, dataset_train, dataset_val, optimizer, scheduler, epochs, device, args_dict
):
    """
    A single function to train `model` on `dataset_train`, validate on `dataset_val`,
    for `epochs` epochs, using `optimizer`, and optionally stepping `scheduler` per batch.
    Logs to Weights & Biases, prints metrics, saves checkpoints.
    """

    machine_name = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["machine_name"] = machine_name
    if args_dict.get("exp_name") is not None:
        exp_name = f"{args_dict['exp_name']}_{timestamp}_{machine_name}"
    else:
        exp_name = f"unnamed_{timestamp}_{machine_name}"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "default_project"),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=exp_name,
        config=args_dict,
    )

    checkpoint_dir = f"model_checkpoints/{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = args_dict["batch_size"]
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, num_workers=4
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 4) Training Loop
    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_accum = 0.0
        num_train_samples = 0
        grad_norms = []

        for X, Y, raw_label_seq in train_loader:
            X = X.to(device)  # shape (batch_size, seq_len, 8)
            Y = Y.to(device)  # shape (batch_size, seq_len)
            optimizer.zero_grad()

            logits = model(X)  # => (batch_size, seq_len, num_classes)
            bsz, seq_len, out_dim = logits.shape

            loss = criterion(logits.view(bsz * seq_len, out_dim), Y.view(bsz * seq_len))
            loss.backward()

            total_norm = 0.0
            max_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm**2
                    if param_norm > max_norm:
                        max_norm = param_norm
            total_norm = total_norm**0.5
            grad_norms.append((total_norm, max_norm))

            optimizer.step()

            train_loss_accum += loss.item() * bsz
            num_train_samples += bsz

            if scheduler is not None:
                scheduler.step()
            global_step += 1

        avg_train_loss = train_loss_accum / num_train_samples

        # Validation
        model.eval()
        val_loss_accum = 0.0
        num_val_samples = 0

        with torch.no_grad():
            for X, Y, raw_label_seq in val_loader:
                X = X.to(device)
                Y = Y.to(device)
                logits = model(X)  # (bsz, seq_len, out_dim)
                bsz, seq_len, out_dim = logits.shape
                vloss = criterion(
                    logits.view(bsz * seq_len, out_dim), Y.view(bsz * seq_len)
                )
                val_loss_accum += vloss.item() * bsz
                num_val_samples += bsz

        avg_val_loss = val_loss_accum / num_val_samples

        current_lr = None
        for param_group in optimizer.param_groups:
            current_lr = param_group["lr"]
            break

        if len(grad_norms) > 0:
            avg_total_norm = sum(g[0] for g in grad_norms) / len(grad_norms)
            avg_max_norm = sum(g[1] for g in grad_norms) / len(grad_norms)
        else:
            avg_total_norm = 0.0
            avg_max_norm = 0.0

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "learning_rate": current_lr,
                "avg_grad_norm": avg_total_norm,
                "max_grad_norm": avg_max_norm,
            },
            step=epoch,
        )

        print(
            f"[Epoch {epoch}] "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"GradNorm(avg/max): {avg_total_norm:.4f}/{avg_max_norm:.4f}"
        )

        current_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        model_info = {
            "model_state_dict": model.state_dict(),
        }
        save_dict = {
            "model_info": model_info,
            "args_dict": args_dict,
        }
        torch.save(save_dict, current_checkpoint_path)


def train_ann(
    model, dataset_train, dataset_val, optimizer, scheduler, epochs, device, args_dict
):
    """
    Single function that trains 'model' on dataset_train, validates on dataset_val,
    logs to wandb, and saves checkpoints every epoch.
    """


    machine_name = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args_dict["machine_name"] = machine_name

    if args_dict["exp_name"] is not None:
        exp_name = f"{args_dict['exp_name']}_{timestamp}_{machine_name}"
    else:
        exp_name = f"unnamed_{timestamp}_{machine_name}"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "default_project"),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=exp_name,
        config=args_dict,
    )

    checkpoint_dir = f"model_checkpoints/{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    batch_size = args_dict["batch_size"]
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False, drop_last=False
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_train = 0

        grad_norm_sum = 0.0
        max_grad_norm = 0.0
        count_batches = 0

        for batch_data in train_loader:
            x, y, _ = batch_data
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2).item()
                    total_norm += param_norm**2
                    if param_norm > max_grad_norm:
                        max_grad_norm = param_norm

            total_norm = total_norm**0.5
            grad_norm_sum += total_norm
            count_batches += 1

            optimizer.step()

            running_loss += loss.item() * x.size(0)
            total_train += x.size(0)

        train_loss = running_loss / total_train
        avg_grad_norm = grad_norm_sum / max(1, count_batches)

        current_lr = None
        if scheduler is not None:
            scheduler.step()
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]
        else:
            for param_group in optimizer.param_groups:
                current_lr = param_group["lr"]

        # Validation loop
        model.eval()
        val_loss = 0.0
        total_val = 0
        with torch.no_grad():
            for batch_data in val_loader:
                x_val, y_val, _ = batch_data
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                logits_val = model(x_val)
                loss_val = criterion(logits_val, y_val)

                val_loss += loss_val.item() * x_val.size(0)
                total_val += x_val.size(0)

        val_loss = val_loss / total_val

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "avg_grad_norm": avg_grad_norm,
                "max_grad_norm": max_grad_norm,
            }
        )

        print(
            f"Epoch [{epoch}/{epochs}] - "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"LR: {current_lr:.6f}, "
            f"AvgGradNorm: {avg_grad_norm:.3f}, "
            f"MaxGradNorm: {max_grad_norm:.3f}"
        )

        current_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        model_info = {
            "model_state_dict": model.state_dict(),
        }
        save_dict = {
            "model_info": model_info,
            "args_dict": args_dict,
        }
        torch.save(save_dict, current_checkpoint_path)

    wandb.finish()


def train_trahgr(
    model, dataset_train, dataset_val, optimizer, scheduler, epochs, device, args_dict
):
    machine_name = socket.gethostname()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args_dict = dict(args_dict)
    args_dict['machine_name'] = machine_name
    if args_dict.get('exp_name') is not None:
        exp_name = f"{args_dict['exp_name']}_{timestamp}_{machine_name}"
    else:
        exp_name = f"unnamed_{timestamp}_{machine_name}"

    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "default_project"),
        entity=os.environ.get("WANDB_ENTITY", None),
        name=exp_name,
        config=args_dict
    )

    checkpoint_dir = f"model_checkpoints/{exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    batch_size = args_dict['batch_size']
    num_workers = args_dict.get('num_workers', 4)
    pin_memory = args_dict.get('pin_memory', torch.cuda.is_available())
    drop_last = args_dict.get('drop_last', False)

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        grad_norm_sum = 0.0
        grad_norm_max_epoch = 0.0
        grad_steps = 0

        for temporal_patches, featural_patches, y, _ in train_loader:
            temporal_patches = temporal_patches.to(device).float()
            featural_patches = featural_patches.to(device).float()
            y = y.to(device).long()

            optimizer.zero_grad(set_to_none=True)

            y_fused, y_tnet, y_fnet = model(temporal_patches, featural_patches)
            loss = criterion(y_tnet, y) + criterion(y_fnet, y) + criterion(y_fused, y)
            loss.backward()

            total_sq = 0.0
            max_g = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                gn = g.norm(2).item()
                total_sq += gn * gn
                if gn > max_g:
                    max_g = gn
            step_gn = math.sqrt(total_sq) if total_sq > 0 else 0.0
            grad_norm_sum += step_gn
            grad_norm_max_epoch = max(grad_norm_max_epoch, max_g)
            grad_steps += 1

            optimizer.step()

            bsz = y.size(0)
            train_loss_sum += loss.item() * bsz
            train_count += bsz

        train_loss = train_loss_sum / max(1, train_count)
        avg_grad_norm = grad_norm_sum / max(1, grad_steps)
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for temporal_patches, featural_patches, y, _ in val_loader:
                temporal_patches = temporal_patches.to(device).float()
                featural_patches = featural_patches.to(device).float()
                y = y.to(device).long()

                y_fused, y_tnet, y_fnet = model(temporal_patches, featural_patches)
                vloss = criterion(y_tnet, y) + criterion(y_fnet, y) + criterion(y_fused, y)

                bsz = y.size(0)
                val_loss_sum += vloss.item() * bsz
                val_count += bsz

        val_loss = val_loss_sum / max(1, val_count)

        if scheduler is not None:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr'] if optimizer.param_groups else None
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': current_lr,
            'avg_grad_norm': avg_grad_norm,
            'max_grad_norm': grad_norm_max_epoch,
        })

        # terminal print
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  "
            f"lr={current_lr:.6g}  avg_grad_norm={avg_grad_norm:.4f}  max_grad_norm={grad_norm_max_epoch:.4f}"
        )

        # checkpoint
        current_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
        model_info = {
            'model_state_dict': model.state_dict(),
        }
        save_dict = {
            'model_info': model_info,
            'args_dict': args_dict,
        }
        torch.save(save_dict, current_checkpoint_path)
