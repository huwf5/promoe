"""
Trace and Model utilities for train_predict_model.py

This module provides:
- Trace: loads and prepares expert selection traces from llama.cpp output
- CustomDataset: PyTorch dataset wrapper
- ModelContext: model training context with loss/optimizer
- SimpleNN: simple feedforward predictor network
"""

import torch
import torch.nn as nn
import os
import glob
import json
from pathlib import Path


def custom_norm_to_max_1(x: torch.Tensor) -> torch.Tensor:
    """Scale gate logits so that, per (..., V) row, max |value| is 1 (last dim = expert)."""
    x = x.float()
    denom = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / denom


def custom_norm_std(x: torch.Tensor) -> torch.Tensor:
    """Per-row z-norm over the expert dimension (last dim)."""
    x = x.float()
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True).clamp(min=1e-8)
    return (x - mean) / std


def custom_norm_fake_gate(x: torch.Tensor) -> torch.Tensor:
    """`input_norm_method=replace`: map logits to a simplex via softmax (stable targets for L1)."""
    return torch.softmax(x.float(), dim=-1)


class Trace:
    """
    Encapsulates expert selection trace data from llama.cpp --trace-logits output.
    
    Trace data structure (from llama.cpp trace dump directory):
    - Prefill stage: per-sequence expert selections
    - Decode stage: per-token expert selections and logits
    
    After unpack_from_dir() + prepare_tensors(), accessible fields:
    - expert_selection: [N_entry, N_layer, K] int64, expert indices per token/layer
    - decode_stage_moe_layer_logits_per_token: [N_decode_token, N_layer+1, H] float32
      hidden features used as predictor input (aligned with cpp kMoeLayerLogits)
    - decode_stage_moe_layer_gate_logits_per_token: [N_decode_token, N_layer, N_expert] float32
    - decode_stage_expert_freq_per_token: [N_decode_token, N_layer, N_expert] float32
    - decode_stage_token_ids_per_token: [N_decode_token] int64
    - decode_stage_seq_id_of_token: [N_decode_token] int64
    - decode_stage_token_idx_in_seq: [N_decode_token] int64
    - decode_stage_token_idx_in_seq_flip: [N_decode_token] int64 (inverse of idx_in_seq within decode)
    - token_ids: [N_decode_token] int64
    - num_expert, num_moe_layer, per_token_expert: inferred metadata
    """
    
    def __init__(self):
        self.expert_selection = None
        
        # Decode stage tensors (main training data source)
        self.decode_stage_moe_layer_logits_per_token = None
        self.decode_stage_moe_layer_gate_logits_per_token = None
        self.decode_stage_expert_freq_per_token = None
        self.decode_stage_token_ids_per_token = None
        self.decode_stage_seq_id_of_token = None
        self.decode_stage_token_idx_in_seq = None
        self.decode_stage_token_idx_in_seq_flip = None
        
        # Prefill stage tensors (less commonly used)
        self.prefill_stage_expert_selection = None
        self.prefill_expert_len = None
        self.prefill_stage_seq_id = None
        
        # Token info
        self.token_ids = None
        
        # Metadata
        self.num_expert = None
        self.num_moe_layer = None
        self.per_token_expert = None
        self.trace_metas = {}
        self.schema_version = 1
        self.id_space = "local"
        self.num_layer = None
        self.num_encoder_moe_layer = 0
        self.num_decoder_moe_layer = None
        self.predict_stage = "decoder"
        self.predictor_target_start = 0
        self.predictor_target_stop = None
        self.predictor_source_ids = None
    
    def unpack_from_dir(self, trace_dir: str):
        """
        Load trace data from directory output by llama.cpp --trace-dump-path.
        
        Expected files in trace_dir:
        - expert_selection.pt: [N_entry, N_layer, K] - decode stage expert choices
        - decode_stage_moe_layer_logits_per_token.pt
        - decode_stage_moe_layer_gate_logits_per_token.pt  
        - decode_stage_expert_freq_per_token.pt
        - decode_stage_token_ids_per_token.pt
        - decode_stage_seq_id_of_token.pt
        - decode_stage_token_idx_in_seq.pt
        - prefill_expert_selection.pt
        - prefill_expert_len.pt (per-layer expert count in prefill for each sequence)
        """
        trace_dir = Path(trace_dir)

        trace_metas_path = trace_dir / "trace_metas.json"
        if trace_metas_path.exists():
            self.trace_metas = json.loads(trace_metas_path.read_text())
        
        # Load decode stage tensors
        self.expert_selection = self._load_pt(trace_dir / "expert_selection.pt")
        
        self.decode_stage_moe_layer_logits_per_token = self._load_pt(
            trace_dir / "decode_stage_moe_layer_logits_per_token.pt"
        )
        self.decode_stage_moe_layer_gate_logits_per_token = self._load_pt(
            trace_dir / "decode_stage_moe_layer_gate_logits_per_token.pt"
        )
        self.decode_stage_expert_freq_per_token = self._load_pt(
            trace_dir / "decode_stage_expert_freq_per_token.pt"
        )
        self.decode_stage_token_ids_per_token = self._load_pt(
            trace_dir / "decode_stage_token_ids_per_token.pt"
        )
        self.decode_stage_seq_id_of_token = self._load_pt(
            trace_dir / "decode_stage_seq_id_of_token.pt"
        )
        self.decode_stage_token_idx_in_seq = self._load_pt(
            trace_dir / "decode_stage_token_idx_in_seq.pt"
        )
        
        # Load prefill stage tensors
        self.prefill_stage_expert_selection = self._load_pt(
            trace_dir / "prefill_expert_selection.pt", required=False
        )
        self.prefill_expert_len = self._load_pt(
            trace_dir / "prefill_expert_len.pt", required=False
        )
        
        # Token IDs (global mapping)
        self.token_ids = self._load_pt(trace_dir / "token_ids.pt", required=False)
        if self.token_ids is None and self.decode_stage_token_ids_per_token is not None:
            self.token_ids = self.decode_stage_token_ids_per_token
    
    def prepare_tensors(self):
        """
        Post-process loaded tensors:
        - Compute decode_stage_token_idx_in_seq_flip (for token_distance filtering)
        - Infer num_expert, num_moe_layer, per_token_expert from expert_selection
        """
        if self.expert_selection is None:
            raise RuntimeError("expert_selection not loaded; call unpack_from_dir first")
        
        # Infer metadata from dense tensors when available. expert_selection may not
        # cover every expert in small or biased traces, but gate/freq last dim is V.
        if self.decode_stage_moe_layer_gate_logits_per_token is not None:
            self.num_expert = self.decode_stage_moe_layer_gate_logits_per_token.shape[-1]
        elif self.decode_stage_expert_freq_per_token is not None:
            self.num_expert = self.decode_stage_expert_freq_per_token.shape[-1]
        else:
            self.num_expert = int(torch.max(self.expert_selection)) + 1
        self.num_moe_layer = self.expert_selection.shape[1]
        self.per_token_expert = self.expert_selection.shape[2]
        self._prepare_trace_metadata()
        
        # Compute token_idx_in_seq_flip: inverse position in sequence during decode
        # This is used for --token_distance filtering (favor tokens later in sequence)
        if self.decode_stage_token_idx_in_seq is not None:
            seq_id = self.decode_stage_seq_id_of_token
            token_idx = self.decode_stage_token_idx_in_seq
            
            # Group by sequence and compute max idx per sequence
            unique_seqs, inverse_indices = torch.unique(seq_id, return_inverse=True)
            max_idx_per_seq = torch.zeros(len(unique_seqs), dtype=token_idx.dtype, device=token_idx.device)
            for i, s in enumerate(unique_seqs):
                mask = seq_id == s
                max_idx_per_seq[i] = torch.max(token_idx[mask])
            
            # Compute flip: max_idx - current_idx
            self.decode_stage_token_idx_in_seq_flip = max_idx_per_seq[inverse_indices] - token_idx

    def _prepare_trace_metadata(self):
        self.schema_version = int(self.trace_metas.get("schema_version", 1))
        self.id_space = self.trace_metas.get("id_space", "local")
        self.predict_stage = self.trace_metas.get("predict_stage", "decoder")

        if self.id_space == "global" and self.predict_stage == "decoder":
            self.num_layer = int(self.trace_metas["num_layer"])
            self.num_encoder_moe_layer = int(self.trace_metas["num_encoder_moe_layer"])
            self.num_decoder_moe_layer = int(self.trace_metas["num_decoder_moe_layer"])
            decoder_start = int(self.trace_metas["decoder_global_layer_start"])
            decoder_stop = int(self.trace_metas["decoder_global_layer_stop"])

            if self.schema_version != 2:
                raise ValueError(f"global decoder trace requires schema_version=2, got {self.schema_version}")
            if self.num_layer != self.num_moe_layer:
                raise ValueError(
                    f"global dense trace num_layer={self.num_layer} does not match "
                    f"expert_selection layer count={self.num_moe_layer}"
                )
            if self.num_encoder_moe_layer + self.num_decoder_moe_layer != self.num_layer:
                raise ValueError(
                    "global decoder trace metadata mismatch: "
                    f"E={self.num_encoder_moe_layer}, D={self.num_decoder_moe_layer}, L={self.num_layer}"
                )
            if decoder_start != self.num_encoder_moe_layer or decoder_stop != self.num_layer:
                raise ValueError(
                    "decoder global range must match encoder boundary and total layer count: "
                    f"start={decoder_start}, stop={decoder_stop}, E={self.num_encoder_moe_layer}, L={self.num_layer}"
                )

            self.predictor_target_start = self.num_encoder_moe_layer
            self.predictor_target_stop = self.num_layer
            self.predictor_source_ids = list(range(self.num_encoder_moe_layer, self.num_layer + 1))
            return

        self.num_layer = self.num_moe_layer
        self.num_encoder_moe_layer = int(self.trace_metas.get("num_encoder_moe_layer", 0))
        self.num_decoder_moe_layer = self.trace_metas.get("num_decoder_moe_layer", self.num_moe_layer)
        if self.num_decoder_moe_layer is not None:
            self.num_decoder_moe_layer = int(self.num_decoder_moe_layer)
        self.predictor_target_start = 0
        self.predictor_target_stop = self.num_moe_layer
        self.predictor_source_ids = list(range(self.num_moe_layer + 1))
    
    @staticmethod
    def _load_pt(path, required=True):
        """Load a .pt file, return None if not found and not required."""
        path = Path(path)
        if path.exists():
            return torch.load(path, map_location='cpu')
        elif required:
            raise FileNotFoundError(f"Expected trace file not found: {path}")
        else:
            return None


class CustomDataset(torch.utils.data.Dataset):
    """Simple dataset wrapper for (input, label, metadata) tuples."""
    
    def __init__(self, input_tensor, label_tensor, metadata_tensor=None):
        self.input = input_tensor
        self.label = label_tensor
        self.metadata = metadata_tensor if metadata_tensor is not None else torch.zeros((len(input_tensor), 0))
        
        assert len(self.input) == len(self.label), \
            f"Input and label length mismatch: {len(self.input)} vs {len(self.label)}"
        assert len(self.input) == len(self.metadata), \
            f"Input and metadata length mismatch: {len(self.input)} vs {len(self.metadata)}"
        
        # Store original labels for evaluation
        self.orig_labels = label_tensor
    
    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, idx):
        return self.input[idx], self.label[idx], self.metadata[idx]
    
    def split(self, train_ratio: float = 0.9):
        """
        Split dataset into train and test.
        
        Args:
            train_ratio: fraction for training (default 90%)
        
        Returns:
            (train_dataset, test_dataset)
        """
        n = len(self)
        n_train = int(n * train_ratio)
        
        indices = torch.randperm(n)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        train_data = CustomDataset(
            self.input[train_indices],
            self.label[train_indices],
            self.metadata[train_indices]
        )
        test_data = CustomDataset(
            self.input[test_indices],
            self.label[test_indices],
            self.metadata[test_indices]
        )
        
        return train_data, test_data
    
    def to(self, device):
        """Move dataset tensors to device."""
        self.input = self.input.to(device)
        self.label = self.label.to(device)
        self.metadata = self.metadata.to(device)
        self.orig_labels = self.orig_labels.to(device)
        return self


class SimpleNN(nn.Module):
    """
    Simple feedforward network for expert frequency prediction.
    
    Args:
        input_size: input feature dimension (e.g., num_expert for layer logits)
        hidden_size: hidden layer dimension
        output_size: output dimension (e.g., num_expert * num_output_layers)
        n_layer: number of hidden layers (default 2)
        dropout: dropout probability (default 0.5)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 n_layer: int = 2, dropout: float = 0.5):
        super().__init__()
        
        layers = []
        
        # Input -> first hidden layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Last hidden layer -> output (no activation)
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


class ModelContext:
    """
    Training context wrapper combining model, loss function, and optimizer.
    
    Args:
        model: torch.nn.Module to train
        loss_class: loss function class (e.g., nn.L1Loss)
        optimizer_class: optimizer class (e.g., torch.optim.Adam)
        lr: learning rate (default 0.001)
        **optimizer_kwargs: additional kwargs for optimizer
    """
    
    def __init__(self, model, loss_class, optimizer_class, lr=0.001, **optimizer_kwargs):
        self.model = model
        self.loss_fn = loss_class(reduction='mean')
        self.optimizer = optimizer_class(model.parameters(), lr=lr, **optimizer_kwargs)
    
    def model_forward_with_loss_and_optimize(self, inputs, labels):
        """Forward pass, compute loss, backward, and step optimizer."""
        self.optimizer.zero_grad()
        
        # Flatten inputs for model
        batch_size = inputs.shape[0]
        inputs_flat = inputs.reshape(batch_size, -1).float()
        
        # Forward
        outputs = self.model(inputs_flat)
        
        # Reshape and compute loss
        outputs_reshaped = outputs.reshape_as(labels)
        loss = self.loss_fn(outputs_reshaped, labels.float())
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return outputs, loss.item()
    
    def model_forward_with_loss(self, inputs, labels):
        """Forward pass and compute loss (no optimization)."""
        with torch.no_grad():
            batch_size = inputs.shape[0]
            inputs_flat = inputs.reshape(batch_size, -1).float()
            outputs = self.model(inputs_flat)
            outputs_reshaped = outputs.reshape_as(labels)
            loss = self.loss_fn(outputs_reshaped, labels.float())
        
        return outputs, loss.item()
    
    def model_forward(self, inputs):
        """Forward pass only (inference)."""
        with torch.no_grad():
            batch_size = inputs.shape[0]
            inputs_flat = inputs.reshape(batch_size, -1).float()
            outputs = self.model(inputs_flat)
        
        return outputs
