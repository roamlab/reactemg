import rospy
import argparse
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import random
import time
from nn_models import Any2Any_Model
from minlora import add_lora, merge_lora, LoRAParametrization
from functools import partial
from collections import deque
from scipy.signal import medfilt

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from myhand_interface import MyHandInterface



class EMGClassification(MyHandInterface):
    def __init__(self, 
                 sync_target,
                 gt_source,
                 saved_checkpoint_pth,
                 lookahead,
                 weight_max_factor,
                 likelihood_format,
                 samples_between_prediction,
                 maj_vote_range,
                 weighting_method
                 ):
        # Shutdown Hook
        rospy.on_shutdown(self.shutdownhook)

        # Read dictionary from provided path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {self.device}')
        checkpoint = torch.load(saved_checkpoint_pth, map_location=self.device)
        self.args_dict = checkpoint['args_dict']

        # Set seeds and parameters
        torch.manual_seed(self.args_dict['seed'])
        random.seed(self.args_dict['seed'])
        np.random.seed(self.args_dict['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.window_size = self.args_dict['window_size']
        self.median_filter_size = self.args_dict['median_filter_size']
        self.embedding_method = self.args_dict['embedding_method']

        # Initialize model
        self.model = Any2Any_Model(
            self.args_dict['embedding_dim'], 
            self.args_dict['nhead'], 
            self.args_dict['dropout'], 
            self.args_dict['activation'], 
            self.args_dict['num_layers'], 
            self.args_dict['window_size'], 
            self.args_dict['embedding_method'],
            self.args_dict['mask_alignment'],
            self.args_dict['share_pe'],
            self.args_dict['tie_weight'],
            self.args_dict['use_input_layernorm'],
            self.args_dict['num_classes'],
            self.args_dict['output_reduction_method'],
            self.args_dict['chunk_size'],
        )

        self.model.load_state_dict(checkpoint['model_info']['model_state_dict'], strict=False)

        self.model.to(self.device)

        self.model.eval()

        # Store number of classes
        # Adding the mask class to the range of possible logits
        self.num_classes = self.args_dict['num_classes'] + 1

        # Circular buffer for incoming EMG data
        self.buffer = np.zeros((self.window_size, 8), dtype=np.float32)
        self.buffer_index = 0

        # Create dummy sequences for the model’s forward pass
        self.dummy_task_idx = torch.tensor([0], device=self.device)
        self.action_mask_token = 3
        self.emg_dummy_mask_positions = torch.tensor(
            np.full((1, self.window_size, 8), 0, dtype=bool),
            device=self.device
        )
        self.action_dummy_sequence = torch.tensor(
            np.full((1, self.window_size,), self.action_mask_token, dtype=np.int64),
            device=self.device
        )

        # Preallocate input tensor
        self.input_tensor = torch.zeros((1, self.window_size, 8), dtype=torch.float32, device=self.device)
        
        # Initialize last prediction
        self.last_prediction = -1  

        # Action mapping for terminal output
        self.action_mapping = {
            0: "Relax",
            1: "Open",
            2: "Close"
        }

        # Number of future windows to consider when makign a prediction for the current timestep
        self.lookahead = lookahead
        # The maximum weight assigned to the farthest future window
        self.weight_max_factor = weight_max_factor
        # 'logits' or 'probs' for the likelihood
        self.likelihood_format = likelihood_format
        # Produce predictions at a lower frequency if specified
        self.samples_between_prediction = samples_between_prediction
        # Range of timestep we consider when doing majority vote
        self.maj_vote_range = maj_vote_range
        # How to aggregate weighted predictions
        self.weighting_method = weighting_method


        # 1) Create a circular buffer for logits
        self.logits_storage = torch.zeros((self.lookahead + 1, self.window_size, self.num_classes),
                                          dtype=torch.float32)
        self.inference_counter = 0
        # Initialize the parent class
        super(EMGClassification, self).__init__(sync_target, gt_source, False)

    def get_emg_data(self, emg_msg):
        # Call the parent class method to store EMG data
        super().get_emg_data(emg_msg)

        # Get sensor data
        x = self.current_sensor_data.get("emg")
        if x is None:
            return None # No EMG data available

        # Append new sample to the buffer
        idx = self.buffer_index % self.window_size
        self.buffer[idx, :] = x
        self.buffer_index += 1

        # Attempt inference immediately if we have enough data
        if self.buffer_index >= self.window_size:
            final_label = self.predict()
            if final_label is not None:
                self.publish_actions(final_label)

    def repeat_chunks(self, tensor_3d: torch.Tensor, original_length: int) -> torch.Tensor:
        """
        Upsamples a reduced output of shape (B, T_red, dim) back to (B, original_length, dim)
        by repeating each timestep equally.

        Assumes that original_length is an integer multiple of T_red.

        Args:
            tensor_3d: shape = (B, T_red, dim)
            original_length: desired upsampled length (e.g. window_size)

        Returns:
            repeated: shape = (B, original_length, dim)
        """
        B, T_red, D = tensor_3d.shape
        chunk_size = original_length // T_red  # integer division

        # Expand along a new dimension, then reshape:
        # (B, T_red, 1, D) -> (B, T_red, chunk_size, D) -> (B, T_red*chunk_size, D)
        repeated = tensor_3d.unsqueeze(2).expand(-1, -1, chunk_size, -1)
        repeated = repeated.reshape(B, T_red * chunk_size, D)
        return repeated

    def predict(self):
        """
        Perform one forward pass with the current window.
        Store the resulting logits, then see if we can finalize a label
        for the timestep (current_inference_index - lookahead).
        """
        # 1) Gather current window data
        data = self.get_buffer_data()

        # 2) Optional median filter
        if self.median_filter_size > 1:
            data = medfilt(data, kernel_size=(self.median_filter_size, 1))

        windowed_data = np.abs(data) / 128.0

        # 3) Convert to torch
        self.input_tensor[0] = torch.from_numpy(windowed_data)

        # 4) Forward pass
        with torch.no_grad():

            # torch.cuda.synchronize()
            # start_time = time.perf_counter()

            _, action_logits = self.model.forward(
                self.input_tensor, 
                self.action_dummy_sequence, 
                self.dummy_task_idx, 
                self.emg_dummy_mask_positions
            )
            # action_logits shape: [1, window_size, num_classes]
            action_logits = action_logits[0]  # shape -> (window_size, num_classes)

            if action_logits.size(0) < self.args_dict['window_size']:
                action_logits = self.repeat_chunks(action_logits,  self.args_dict['window_size'])

            # torch.cuda.synchronize()
            # inference_time = time.perf_counter() - start_time
            # print(f"Inference time: {inference_time*1000:.3f} ms")

        # Store in circular buffer
        self.logits_storage[self.inference_counter % (self.lookahead + 1)] = action_logits.cpu()

        current_time_index = self.inference_counter
        self.inference_counter += 1

        # Check if we can finalize a label for (current_time_index - lookahead)
        final_time = current_time_index - self.lookahead
        if final_time < 0:
            return None

        # Only produce a label every 'samples_between_prediction' steps
        if (current_time_index % self.samples_between_prediction) != 0:
            return None

        # If we are due for a prediction, aggregate logits
        final_label = self.aggregate_logits(final_time)
        return final_label

    def aggregate_logits(self, t):
        """
        Weighted average of logits for physical time t,
        using the windows t..t+lookahead (where i-th window covers real times [i-window_size+1 .. i]).
        We only take rows (timesteps) >= t within each window.
        The weight for each future index i is linearly scaled from 1.0 to self.weight_max_factor.
        If self.maj_vote_range == 'future', we sum from offset..end of each window.
        If self.maj_vote_range == 'single', we only sum the single row at offset (i.e. time t).
        """
        sum_logits = torch.zeros(self.num_classes, dtype=torch.float32)
        total_weight = 0.0

        for i in range(t, t + self.lookahead):
            window_logits = self.logits_storage[i % (self.lookahead)]  # (window_size, num_classes)
            start_time = i - self.window_size + 1

            offset = t - start_time

            if offset < 0:
                offset = 0
            elif offset >= self.window_size:
                continue

            # Decide on what range of data to do majority vote on base on maj_vote_range
            if self.maj_vote_range == "future":
                relevant_chunk = window_logits[offset:, :]
            else:
                relevant_chunk = window_logits[offset:offset+1, :]

            # Convert to probabilities if user specified "probs"
            if self.likelihood_format == "probs":
                relevant_chunk = relevant_chunk.softmax(dim=-1)

            # Weighting mechanism
            if self.weighting_method == "equal":
                # Original "equal" weighting (one weight per window i)
                future_distance = i - t
                if self.lookahead > 0:
                    w_i = 1.0 + (self.weight_max_factor - 1.0) * (future_distance / self.lookahead)
                else:
                    w_i = 1.0

                chunk_sum = relevant_chunk.sum(dim=0)
                sum_logits += chunk_sum * w_i
                total_weight += relevant_chunk.size(0) * w_i

            else:  # "progressive" weighting (row-by-row based on local timestep)
                # relevant_chunk.shape is (N, num_classes), where N = number of rows from offset onward
                # Each row r corresponds to time (t + r) if offset == 0, etc.
                chunk_len = relevant_chunk.size(0)
                for r in range(chunk_len):
                    if self.lookahead > 0:
                        w_r = 1.0 + (self.weight_max_factor - 1.0) * (r / self.lookahead)
                    else:
                        w_r = 1.0

                    sum_logits += relevant_chunk[r, :] * w_r
                    total_weight += w_r

        avg_logits = sum_logits / total_weight
        final_label = torch.argmax(avg_logits).item()
        return final_label

    def publish_actions(self, prediction):
        """Publish actions to the hardware based on the prediction."""
        if prediction != self.last_prediction:
            self.pub.publish(prediction)
            self.last_prediction = prediction
            print(self.action_mapping.get(prediction, f"Class {prediction}"))

    def get_buffer_data(self):
        """Retrieve data from the circular buffer in correct time order."""
        idx = self.buffer_index % self.window_size
        if idx == 0:
            data = self.buffer.copy()
        else:
            data = np.concatenate((self.buffer[idx:], self.buffer[:idx]), axis=0)
        return data

    def shutdownhook(self):
        print('Shutting down...')
        self.publish_actions(0)  # e.g. 'Relax' or 'Stop'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_checkpoint_pth", type=str, required=True, 
                        help="checkpoint path to use")
    parser.add_argument("--sync_target", type=str, default="myo", 
                        help="sync target: 'myo' or 'bci'")
    parser.add_argument("--gt_source", type=str, default="button", 
                        help="GT source, e.g. 'button'")
    parser.add_argument("--lookahead", type=int, default=40,
                        help="Number of future timesteps to wait for before finalizing label. "
                             "If > 0, we finalize label at time t only after reaching t+lookahead.")
    parser.add_argument("--weight_max_factor", type=float, default=1.5,
                        help="Max weight assigned to the farthest future window; "
                             "weights linearly ramp from 1.0 (at t) to weight_max_factor (at t+lookahead).")
    parser.add_argument("--likelihood_format", type=str, default="probs",
                        choices=["logits", "probs"],
                        help="Whether to average over raw logits or first convert to probabilities.")
    parser.add_argument("--samples_between_prediction", type=int, default=1,
                        help="Produce a final prediction only every N samples (>=1).")  
    parser.add_argument("--maj_vote_range", type=str, default="future",
                        choices=["single", "future"],
                        help="If 'future', aggregates all rows >= t in each window; "
                             "if 'single', only uses the row for time t in each window.")
    parser.add_argument("--weighting_method", type=str, default="equal",
                        choices=["equal", "progressive"],
                        help="If 'equal', use the original single weight per future window. "
                             "If 'progressive', assign weights based on each row's local timestep.")

    args = parser.parse_args()

    # Ensure samples_between_prediction is valid
    if args.samples_between_prediction < 1:
        raise ValueError("--samples_between_prediction must be >= 1")

    rospy.init_node("realtime_listener", anonymous=True)

    interface = EMGClassification(
        sync_target=args.sync_target,
        gt_source=args.gt_source,
        saved_checkpoint_pth=args.saved_checkpoint_pth,
        lookahead=args.lookahead,
        weight_max_factor=args.weight_max_factor,
        likelihood_format=args.likelihood_format,
        samples_between_prediction=args.samples_between_prediction,
        maj_vote_range=args.maj_vote_range,
        weighting_method=args.weighting_method
    )

    print('Initializing buffer...')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down...")
