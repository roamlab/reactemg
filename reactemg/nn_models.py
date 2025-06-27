import torch
import torch.nn as nn
import torch.nn.functional as F


##############################
############# ANN ############
##############################
class ANN_Model(nn.Module):
    """
    A 4-layer MLP with 1000 neurons in each layer, dropout=0.3, batchnorm, ReLU
    from the paper's best configuration. Input=48 features, output=args.num_classes.
    """

    def __init__(self, num_classes):
        super(ANN_Model, self).__init__()

        self.input_dim = 48
        self.hidden_dim = 1000
        self.num_classes = num_classes

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.drop2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn3 = nn.BatchNorm1d(self.hidden_dim)
        self.drop3 = nn.Dropout(p=0.3)

        self.fc4 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn4 = nn.BatchNorm1d(self.hidden_dim)
        self.drop4 = nn.Dropout(p=0.3)

        self.out = nn.Linear(self.hidden_dim, self.num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: shape [batch_size, 48] (features from dataset)
        returns: shape [batch_size, num_classes]
        """

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)

        x = self.fc4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.drop4(x)

        x = self.out(x)
        return x


################################
############# LSTM #############
################################
class LSTM_Model(nn.Module):
    def __init__(self, input_size=8, fc_size=400, hidden_size=256, num_classes=6):
        """
        input_size : # of input channels per timestep (8 if 8-channel EMG).
        fc_size    : # of neurons in first fully-connected projection (paper used 400).
        hidden_size: # of LSTM hidden units (paper used 256).
        num_classes: number of gesture classes (3,6,...).
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, fc_size)
        self.lstm = nn.LSTM(
            input_size=fc_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, input_size=8)
        Returns logits of shape: (batch_size, seq_len, num_classes)
        """
        batch_size, seq_len, _ = x.shape
        x = torch.tanh(self.fc1(x))  # => (batch_size, seq_len, fc_size)
        x, (h_n, c_n) = self.lstm(x)  # => (batch_size, seq_len, hidden_size)
        logits = self.fc2(x)  # => (batch_size, seq_len, num_classes)
        return logits


##################################
############# ED_TCN #############
##################################
class EDTCN_Model(nn.Module):
    """
    2-layer encoder-decoder TCN for sequence labeling:
      - Each 'encoder' layer:
         1D Conv -> ReLU -> MaxPool (kernel_size=2, stride=2)
      - Each 'decoder' layer:
         Upsample (factor=2) -> 1D Conv -> ReLU
      - Final layer:
         Time-distributed linear -> outputs (B, T, num_classes)
    """

    def __init__(
        self, num_channels=8, num_classes=10, enc_filters=(128, 288), kernel_size=25
    ):
        super().__init__()

        # -------------- Encoder --------------
        # First encoder layer: from (num_channels) to enc_filters[0]
        self.encoder_conv1 = nn.Conv1d(
            in_channels=num_channels,
            out_channels=enc_filters[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )  # 'same' padding
        self.encoder_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Second encoder layer: from enc_filters[0] to enc_filters[1]
        self.encoder_conv2 = nn.Conv1d(
            in_channels=enc_filters[0],
            out_channels=enc_filters[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.encoder_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # -------------- Decoder --------------
        # Mirror the encoder structure but in reverse
        # Do upsampling + 1D conv -> ReLU
        self.decoder_upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder_conv2 = nn.Conv1d(
            in_channels=enc_filters[1],
            out_channels=enc_filters[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        self.decoder_upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.decoder_conv1 = nn.Conv1d(
            in_channels=enc_filters[1],
            out_channels=enc_filters[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Final time-distributed output layer
        self.final_conv = nn.Conv1d(
            in_channels=enc_filters[0], out_channels=num_classes, kernel_size=1
        )  # 1x1 conv over time => (B, num_classes, T)

    def forward(self, x):
        """
        x: shape (B, T, num_channels)
        returns: shape (B, T, num_classes)
        """
        # (B, T, C_in) -> rearrange to (B, C_in, T) for Conv1d
        x = x.transpose(1, 2)  # shape = (B, C_in=8, T=19)

        # --------- Encoder ---------
        # Layer 1
        x = self.encoder_conv1(x)  # (B, f1, T)
        x = F.relu(x)
        x = self.encoder_pool1(x)  # (B, f1, T/2)

        # Layer 2
        x = self.encoder_conv2(x)  # (B, f2, T/2)
        x = F.relu(x)
        x = self.encoder_pool2(x)  # (B, f2, T/4)  [approx if T isn't multiple of 4]

        # --------- Decoder ---------
        # Mirror Layer 2
        x = self.decoder_upsample2(x)  # (B, f2, T/2)
        x = self.decoder_conv2(x)  # (B, f2, T/2)
        x = F.relu(x)

        # Mirror Layer 1
        x = self.decoder_upsample1(x)  # (B, f2, T)
        x = self.decoder_conv1(x)  # (B, f1, T)
        x = F.relu(x)

        # Final conv => (B, num_classes, T)
        x = self.final_conv(x)

        # Permute back => (B, T, num_classes)
        x = x.transpose(
            1, 2
        )  # for a typical cross-entropy with dimension (B, #class, T), adapt below
        return x


###################################
############# Any2Any #############
###################################
class Any2Any_Model(nn.Module):
    def __init__(
        self,
        embedding_dim,
        nhead,
        dropout,
        activation,
        num_layers,
        window_size,
        embedding_method,
        mask_alignment,
        share_pe,
        tie_weight,
        use_decoder,
        use_input_layernorm,
        num_classes,
        output_reduction_method,
        chunk_size,
        inner_window_size,
        use_mav_for_emg,
        mav_inner_stride,
    ):
        super(Any2Any_Model, self).__init__()

        self.embedding_dim = embedding_dim
        self.embedding_method = embedding_method
        self.nhead = nhead
        self.mask_alignment = mask_alignment
        self.use_decoder = use_decoder
        self.use_input_layernorm = use_input_layernorm
        self.output_reduction_method = output_reduction_method
        self.chunk_size = chunk_size
        self.inner_window_size = inner_window_size
        self.use_mav_for_emg = use_mav_for_emg
        self.mav_inner_stride = mav_inner_stride

        self.original_window_size = window_size
        if use_mav_for_emg == 1:
            # Compute how many MAV subwindows per outer window
            # M = floor((outer_window_size - inner_window_size) / mav_inner_stride) + 1
            effective_mav_length = (
                window_size - inner_window_size
            ) // mav_inner_stride + 1
            # Override model's window_size
            self.window_size = effective_mav_length
            # Force coarse=False so we do the normal "dense" path
            self.use_coarse = False
        else:
            # If not using MAV, keep the normal logic
            self.window_size = window_size
            self.use_coarse = inner_window_size < window_size

        # Action vocab + embedding
        self.action_vocab_size = num_classes + 1  # same as before
        self.action_embedding = nn.Embedding(self.action_vocab_size, embedding_dim)

        # Modality-specific embedding for actions
        self.action_modality_specific_embedding = nn.Parameter(
            torch.empty(1, 1, embedding_dim)
        )
        nn.init.uniform_(self.action_modality_specific_embedding, a=-0.02, b=0.02)

        # Transformer encoder definition (same in both modes)
        # Pre-LN
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        # Action output projection
        self.action_output_projection = nn.Linear(embedding_dim, self.action_vocab_size)

        # For coarse mode, we do learned downsample & upsample for EMG
        if self.use_coarse:
            # e.g.  window_size = 100, inner_window_size = 10 => coarse_length=10
            self.coarse_length = self.window_size // self.inner_window_size

            # Downsample + upsample
            self.downsample_emg = nn.Conv1d(
                in_channels=8,
                out_channels=embedding_dim,
                kernel_size=self.inner_window_size,
                stride=self.inner_window_size,
                padding=0
            )
            self.upsample_emg = nn.ConvTranspose1d(
                in_channels=embedding_dim,
                out_channels=8,
                kernel_size=self.inner_window_size,
                stride=self.inner_window_size,
                padding=0
            )

            # Learned EMG mask token at coarse level
            self.linear_projection_learnable_mask_coarse = nn.Parameter(
                torch.empty(1, 1, embedding_dim)
            )
            nn.init.uniform_(
                self.linear_projection_learnable_mask_coarse, a=-0.02, b=0.02
            )

            # Modality-specific embedding for EMG
            self.emg_modality_specific_embedding = nn.ParameterDict()
            self.emg_modality_specific_embedding["emg_id_embedding"] = nn.Parameter(
                torch.empty(1, 1, embedding_dim)
            )
            nn.init.uniform_(
                self.emg_modality_specific_embedding["emg_id_embedding"],
                a=-0.02,
                b=0.02,
            )

            # Positional encodings (coarse_length)
            if share_pe:
                self.emg_positional_encoding = nn.Parameter(
                    torch.empty(1, self.coarse_length, embedding_dim)
                )
                nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
                # Share with action
                self.action_positional_encoding = self.emg_positional_encoding
            else:
                self.emg_positional_encoding = nn.Parameter(
                    torch.empty(1, self.coarse_length, embedding_dim)
                )
                self.action_positional_encoding = nn.Parameter(
                    torch.empty(1, self.coarse_length, embedding_dim)
                )
                nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
                nn.init.uniform_(self.action_positional_encoding, a=-0.02, b=0.02)

        else:
            # ============ Fine-resolution ==============
            self.emg_embedding = nn.Conv1d(8, embedding_dim, kernel_size=1, stride=1)
            self.linear_projection_learnable_mask = nn.Parameter(
                torch.empty(1, embedding_dim, 1)
            )
            nn.init.uniform_(self.linear_projection_learnable_mask, a=-0.02, b=0.02)

            if self.use_input_layernorm:
                self.emg_in_layer_norm = nn.LayerNorm(embedding_dim)

            # Modality-specific embedding for EMG
            self.emg_modality_specific_embedding = nn.ParameterDict()
            self.emg_modality_specific_embedding["emg_id_embedding"] = nn.Parameter(
                torch.empty(1, 1, embedding_dim)
            )
            nn.init.uniform_(
                self.emg_modality_specific_embedding["emg_id_embedding"],
                a=-0.02,
                b=0.02,
            )

            # Positional encoding of size window_size
            if share_pe:
                self.emg_positional_encoding = nn.Parameter(
                    torch.empty(1, self.window_size, embedding_dim)
                )
                nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
                self.action_positional_encoding = self.emg_positional_encoding
            else:
                self.emg_positional_encoding = nn.Parameter(
                    torch.empty(1, self.window_size, embedding_dim)
                )
                self.action_positional_encoding = nn.Parameter(
                    torch.empty(1, self.window_size, embedding_dim)
                )
                nn.init.uniform_(self.emg_positional_encoding, a=-0.02, b=0.02)
                nn.init.uniform_(self.action_positional_encoding, a=-0.02, b=0.02)

            # EMG output projection
            self.emg_output_projection = nn.Conv1d(
                embedding_dim, 8, kernel_size=1, stride=1
            )

        # Tie action embedding with action output projection if specified
        if tie_weight:
            self.action_output_projection.weight = self.action_embedding.weight
            pass

        # Possibly add aggregator for chunk-based output, as in
        if self.output_reduction_method == "learned":
            self.chunk_aggregator = nn.Linear(
                self.chunk_size * embedding_dim, embedding_dim
            )
        else:
            self.chunk_aggregator = None

    def _reduce_sequence_by_pooling(self, x, chunk_size):
        """
        x: (B, T, d_model)
        Pool in time dimension with chunk_size
        """
        B, T, D = x.size()
        # Assume T % chunk_size == 0
        x = x.reshape(B, T // chunk_size, chunk_size, D)  # (B, T//cs, cs, D)
        return x.mean(dim=2)  # (B, T//cs, D)

    def _reduce_sequence_by_learned(self, x, chunk_size, aggregator):
        """
        x: (B, T, d_model)
        Flatten each chunk, then apply aggregator
        """
        B, T, D = x.size()
        x = x.reshape(B, T // chunk_size, chunk_size * D)  # (B, T//cs, cs*D)
        return aggregator(x)  # (B, T//cs, d_model)

    def forward(
        self,
        masked_emg,
        masked_actions,
        task_idx,
        mask_positions_emg,
        return_output=False,
        emg_window=None,
        action_window=None,
    ):
        """
        masked_emg, masked_actions => Typically shape is either:
          - If not coarse: (B, window_size, 8) and (B, window_size)
          - If coarse: pass them as (B, coarse_length, 8) or so, depending on the DataLoader,
            but the recommended approach (per the plan) is that the dataset does *not* do actual numeric masking on EMG,
            so `masked_emg` can be the raw or partially masked shape, and we do the real "embedding + mask" inside here.
        """
        batch_size = masked_emg.size(0)

        # ----------------- Main pipeline (dense) -----------------
        if not self.use_coarse:
            # linear projection
            masked_emg = self.emg_embedding(
                masked_emg.transpose(1, 2)
            )  # (B, embed_dim, W)
            if self.use_input_layernorm:
                masked_emg = self.emg_in_layer_norm(masked_emg.transpose(1, 2))
                masked_emg = masked_emg.transpose(1, 2)  # (B, embed_dim, W) again

            # Apply mask tokens if aligned
            if self.mask_alignment == "non-aligned":
                raise Exception(
                    "non-aligned not implemented here for linear_projection"
                )
            elif self.mask_alignment == "aligned":
                # expand to (B, embed_dim, W)
                mask_tokens = self.linear_projection_learnable_mask.expand(
                    masked_emg.shape[0], -1, masked_emg.shape[2]
                )
                mask_positions_emg_t = mask_positions_emg.transpose(1, 2)

                # Use channel 0's mask position is sufficient in the case where the mask positions are aligned.
                zeroth_embedding_values = mask_positions_emg_t[:, 0, :].unsqueeze(1)
                expanded_mask_positions = zeroth_embedding_values.expand(
                    -1, self.embedding_dim, -1
                ).type_as(mask_tokens)
                masked_emg = (
                    masked_emg * (1.0 - expanded_mask_positions)
                    + mask_tokens * expanded_mask_positions
                )
            else:
                raise Exception(f"Unrecognized mask_alignment: {self.mask_alignment}")

            # Add modality-specific embedding & pos encoding
            masked_emg = masked_emg.transpose(1, 2)  # => (B, W, embed_dim)
            masked_emg = (
                masked_emg
                + self.emg_modality_specific_embedding["emg_id_embedding"]
                + self.emg_positional_encoding
            )

            # Action embedding
            action_embedded = (
                self.action_embedding(masked_actions)
                + self.action_modality_specific_embedding
                + self.action_positional_encoding
            )

            # Concatenate
            src = torch.cat([masked_emg, action_embedded], dim=1)
            seq_len = src.size(1)  # should be 2 * self.window_size

            # 6) ### ATTENTION MASK ### in the fine-resolution branch
            attention_mask = torch.zeros(
                (batch_size, seq_len, seq_len), dtype=torch.bool, device=src.device
            )
            # If task_idx == 3 => block all action positions
            # “action positions” here are the last window_size tokens
            for b in range(batch_size):
                if task_idx[b] == 3:
                    # block [all queries from action, all keys from action]
                    attention_mask[b, :, -self.window_size :] = True
                    attention_mask[b, -self.window_size :, :] = True

            # Expand mask for multihead
            attention_mask = attention_mask.repeat_interleave(self.nhead, dim=0)

            # Transformer
            src = self.transformer_encoder(src, mask=attention_mask)

            if return_output:
                return src

            # Split back
            emg_encoded = src[:, : self.window_size, :]
            action_encoded = src[:, self.window_size :, :]

            # Project EMG
            emg_output = self.emg_output_projection(emg_encoded.transpose(1, 2))
            emg_output = emg_output.transpose(1, 2)  # => (B, W, 8)

            # Project Action
            if self.output_reduction_method == "none":
                action_output = self.action_output_projection(action_encoded)
            elif self.output_reduction_method == "pooling":
                pooled = self._reduce_sequence_by_pooling(
                    action_encoded, self.chunk_size
                )
                action_output = self.action_output_projection(pooled)
            elif self.output_reduction_method == "learned":
                learned_agg = self._reduce_sequence_by_learned(
                    action_encoded, self.chunk_size, self.chunk_aggregator
                )
                action_output = self.action_output_projection(learned_agg)
            else:
                raise ValueError(
                    f"Unknown output_reduction_method: {self.output_reduction_method}"
                )

            return emg_output, action_output

        # ----------------- IF COARSE: new downsample/upsample path -----------------
        else:
            # masked_emg is presumably the raw (B, window_size, 8), unmasked
            # downsample
            emg_down = self.downsample_emg(masked_emg.transpose(1, 2))
            # => shape (B, embedding_dim, coarse_length)
            emg_down = emg_down.transpose(1, 2)
            # => shape (B, coarse_length, embedding_dim)

            # Replace masked sub-window vectors with learned mask token
            # mask_positions_emg is shape (B, coarse_length) boolean
            mask_token = (
                self.linear_projection_learnable_mask_coarse
            )  # shape (1, 1, embedding_dim)
            mask_token = mask_token.expand(
                emg_down.size(0), -1, -1
            )  # => (B, 1, embedding_dim)

            for b in range(batch_size):
                masked_indices = torch.where(mask_positions_emg[b])[0]
                if len(masked_indices) > 0:
                    emg_down[b, masked_indices, :] = mask_token[b, 0, :]

            # Add EMG modality embedding + pos enc
            emg_down = (
                emg_down
                + self.emg_modality_specific_embedding["emg_id_embedding"]
                + self.emg_positional_encoding
            )

            # Action embedding (coarse) => shape (B, coarse_length, embedding_dim)
            action_embedded = (
                self.action_embedding(masked_actions)
                + self.action_modality_specific_embedding
                + self.action_positional_encoding
            )

            # Concatenate => (B, 2*coarse_length, embedding_dim)
            src = torch.cat([emg_down, action_embedded], dim=1)

            # Build attention mask (B, 2*coarse_length, 2*coarse_length)
            # same logic as before but for coarse_length
            seq_len = src.size(1)  # should be 2 * self.coarse_length
            attention_mask = torch.zeros(
                (batch_size, seq_len, seq_len), dtype=torch.bool, device=src.device
            )
            for b in range(batch_size):
                if task_idx[b] == 3:
                    # block last coarse_length entries if they correspond to action
                    attention_mask[b, :, -self.coarse_length :] = True
                    attention_mask[b, -self.coarse_length :, :] = True
            attention_mask = attention_mask.repeat_interleave(self.nhead, dim=0)

            # Transformer
            src = self.transformer_encoder(src, mask=attention_mask)
            if return_output:
                return src

            # Split back: first self.coarse_length => EMG, second => Action
            emg_encoded = src[:, : self.coarse_length, :]
            action_encoded = src[:, self.coarse_length :, :]

            # Upsample EMG => (B, 8, window_size)
            emg_output = self.upsample_emg(emg_encoded.transpose(1, 2))
            # => (B, 8, coarse_length * stride) = (B, 8, window_size)
            emg_output = emg_output.transpose(1, 2)  # => (B, window_size, 8)

            # Action projection => (B, coarse_length, vocab_size)
            if self.output_reduction_method == "none":
                action_output = self.action_output_projection(action_encoded)
            elif self.output_reduction_method == "pooling":
                # chunk/pool over coarse_length. Not used when we're already at the coarse resolution
                pooled = self._reduce_sequence_by_pooling(
                    action_encoded, self.chunk_size
                )
                action_output = self.action_output_projection(pooled)
            elif self.output_reduction_method == "learned":
                learned_agg = self._reduce_sequence_by_learned(
                    action_encoded, self.chunk_size, self.chunk_aggregator
                )
                action_output = self.action_output_projection(learned_agg)
            else:
                raise ValueError(
                    f"Unknown output_reduction_method: {self.output_reduction_method}"
                )

            return emg_output, action_output
