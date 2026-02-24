"""
SMART Components - Minimax Version
===================================
Physics/mathematics-inspired regularizers for neural network training.
Sanitized version for public release.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 1. Entropic Regularizer (Optimized: log_softmax)
# =============================================================================
class EntropicRegularizer(nn.Module):
    """
    Apply entropic regularization to prevent mode collapse and encourage diversity.
    Scales entropy penalty inversely with "knowledge mass" (estimated certainty).
    Optimization: Uses log_softmax for numerical stability and speed.
    """
    def __init__(self, d_model, entropy_scale=0.1):
        super().__init__()
        self.entropy_scale = entropy_scale
        
        # Knowledge mass estimator: how much "information" is in the latent
        self.knowledge_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Learnable entropy temperature
        self.temp_scheduler = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, logits, latent_sequence, mask=None):
        """
        logits: [B, T, V]
        latent_sequence: [B, T, D]
        mask: [B, T] (1 for valid, 0 for pad)
        """
        # Estimate Knowledge Mass
        knowledge_scores = self.knowledge_estimator(latent_sequence)  # [B, T, 1]
        
        if mask is not None:
             # Apply mask to knowledge scores
             mask_expanded = mask.unsqueeze(-1) # [B, T, 1]
             knowledge_scores = knowledge_scores * mask_expanded
             knowledge_mass = knowledge_scores.sum(dim=[1, 2]) / (mask_expanded.sum(dim=[1,2]) + 1e-8)
        else:
             knowledge_mass = knowledge_scores.mean(dim=[1, 2])  # [B]
        
        # Compute Entropy (Optimized)
        temp = self.temp_scheduler.clamp(min=0.1).float()
        # Use float32 for stability
        logits_fp32 = logits.float()
        
        # OPTIMIZATION: log_softmax avoids softmax+log and lets entropy be computed as -sum(p * log p).
        log_probs = F.log_softmax(logits_fp32 / temp, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)  # [B, T]
        
        if mask is not None:
            entropy = (entropy * mask).sum(dim=-1) / (mask.sum(dim=-1) + 1e-8) # [B]
        else:
            entropy = entropy.mean(dim=-1)
            
        # Scale Entropy Penalty
        # Higher knowledge mass -> lower penalty (allow certainty)
        # Keep the scale path in fp32 to match the entropy path and reduce cast churn.
        mean_mass = knowledge_mass.float().mean().clamp(min=1e-6) # Clamp to avoid explosion
        scaled_entropy = entropy.mean() / mean_mass
        scaled_entropy = torch.clamp(scaled_entropy, max=100.0)
        
        return self.entropy_scale * scaled_entropy, entropy.mean(), mean_mass

# =============================================================================
# 2. Holographic Depth Regularizer (Optimized: eigvalsh)
# =============================================================================
class HolographicDepthRegularizer(nn.Module):
    """
    Constraints network depth dynamics to respect holographic bounds (AdS/CFT).
    Optimization: Uses eigvalsh on Gram matrix (symmetric) instead of SVD.
    """
    def __init__(self, n_layers, d_model):
        super().__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        
        # Target Profile: Information density should decrease as 1/depth
        z_coords = torch.linspace(1.0, 10.0, n_layers)
        self.register_buffer('target_entropy_profile', 1.0 / z_coords)
            
    def compute_layer_entropy(self, hidden_state, mask=None):
        # hidden_state: [B, T, D]
        
        B = hidden_state.shape[0]
        if B < 2:
             return torch.tensor(0.0, device=hidden_state.device)
             
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            valid_tokens = hidden_state * mask_expanded
            curr_state = valid_tokens.sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8) # [B, D]
        else:
            curr_state = hidden_state.mean(dim=1) 
        
        # Normalize features per sample
        curr_state = F.normalize(curr_state, dim=1) # [B, D]
        
        gram = torch.matmul(curr_state, curr_state.T) # [B, B]
        
        # Singular values of batch covariance
        # OPTIMIZATION: Gram matrix is symmetric PSD -> Eigenvalues = Singular Values
        # eigvalsh is significantly faster than svdvals for symmetric matrices
        try:
             s = torch.linalg.eigvalsh(gram.float()) 
             # Eigvalsh returns in ascending order, can have small negatives due to precision
             s = s.clamp(min=1e-8)
             
             # Normalize to probability
             p = s / (s.sum() + 1e-8)
             # Shannon entropy of the spectrum
             entropy = -torch.sum(p * torch.log(p + 1e-8))
             return entropy
        except Exception as exc:
             logging.warning("HolographicDepthRegularizer Eig/SVD failed: %s", exc)
             return torch.tensor(0.0, device=hidden_state.device)

    def forward(self, hidden_states, mask=None):
        """
        hidden_states: Tuple of tensors from model output (one per layer)
        """
        # Slice to match n_layers
        if len(hidden_states) > self.n_layers:
            states_to_check = hidden_states[-self.n_layers:]
        else:
            states_to_check = hidden_states
            
        layer_entropies = []
        for h in states_to_check:
            layer_entropies.append(self.compute_layer_entropy(h, mask))
            
        measured_profile = torch.stack(layer_entropies)
        
        # Buffer moves with the module, so it's already on the correct device.
        target = self.target_entropy_profile[: measured_profile.shape[0]]
        
        measured_norm = measured_profile / (measured_profile.max() + 1e-8)
        
        loss = F.mse_loss(measured_norm, target)
        return loss

# =============================================================================
# 3. Differentiable Topology Regularizer (Optimized: cdist)
# =============================================================================
class DifferentiableTopologyRegularizer(nn.Module):
    """
    Soft constraints to encourage connected components and avoid holes in latent space.
    Optimization: Uses torch.cdist for distance matrix calculation.
    """
    def __init__(self, d_model, connectivity_weight=1.0, hole_weight=0.5):
        super().__init__()
        self.connectivity_weight = connectivity_weight
        self.hole_weight = hole_weight
        self.connection_threshold = nn.Parameter(torch.tensor(1.0))

    def forward(self, latent_batch, mask=None, lengths=None):
        if latent_batch.dim() != 3:
            return torch.tensor(0.0, device=latent_batch.device)
             
        B, T, D = latent_batch.shape
        # Always keep these as tensors so downstream callers can safely .item() them.
        total_connectivity_loss = torch.tensor(0.0, device=latent_batch.device)
        total_hole_loss = torch.tensor(0.0, device=latent_batch.device)
        
        for b in range(B):
            sequence = latent_batch[b] # [T, D]
             
            if lengths is not None:
                # Prefer CPU-computed lengths to avoid GPU sync from `.item()`.
                valid_len = int(lengths[b])
                if valid_len < 3: continue
                sequence = sequence[:valid_len]
                T_curr = valid_len
            elif mask is not None:
                valid_len = mask[b].sum().long().item()
                if valid_len < 3: continue
                sequence = sequence[:valid_len]
                T_curr = valid_len
            else:
                T_curr = T
                if T_curr < 3: continue
            
            # Optimization: Stride
            if T_curr > 128:
                stride = T_curr // 128
                sequence = sequence[::stride]
                T_curr = sequence.shape[0]

            if T_curr < 3: continue
            
            # OPTIMIZATION: cdist avoids unsqueeze memory explosion
            # [T, T] distance matrix
            distances = torch.cdist(sequence, sequence, p=2)
            
            # 1. Connectivity
            threshold = torch.abs(self.connection_threshold) + 0.1
            connection_prob = torch.sigmoid(-(distances - threshold))
            
            # Avoid allocating a dense diag mask: sum off-diagonal entries directly.
            connection_sum = connection_prob.sum() - connection_prob.diagonal().sum()
            denom = (T_curr * (T_curr - 1)) + 1e-8
            connectivity_loss = 1 - (connection_sum / denom)
            
            # 2. Hole Avoidance
            hole_loss = torch.tensor(0.0, device=latent_batch.device)
            
            # Standard Random Triplet Sampling (Non-Amortized for Phase 1 Safety)
            n_triplets = min(32, T_curr * (T_curr-1) * (T_curr-2) // 6)
            
            if n_triplets > 0:
                idx = torch.randint(0, T_curr, (n_triplets, 3), device=latent_batch.device)
                d01 = distances[idx[:,0], idx[:,1]]
                d02 = distances[idx[:,0], idx[:,2]]
                d12 = distances[idx[:,1], idx[:,2]]
                
                edges = torch.stack([d01, d02, d12], dim=1) # [N, 3]
                edge_variance = torch.var(edges, dim=1) # [N]
                hole_loss = torch.exp(-edge_variance).mean()
            
            total_connectivity_loss += connectivity_loss
            total_hole_loss += hole_loss
            
        avg_loss = (
            self.connectivity_weight * (total_connectivity_loss / B) + 
            self.hole_weight * (total_hole_loss / B)
        )
        return avg_loss

# =============================================================================
# 4. Manifold Sinkhorn Projection (Standard)
# =============================================================================
class ManifoldRegularizer(nn.Module):
    """
    Penalizes weight matrices that deviate from the doubly-stochastic manifold.
    Standard Version (Processes all layers every step).
    """
    def __init__(self, model, weight_decay=1e-4):
        super().__init__()
        self.weight_decay = weight_decay
        # Identify linear layers to regularize
        self.target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if 'proj' in name:
                    self.target_modules.append(module)
                    
    def forward(self):
        loss = 0.0
        for mod in self.target_modules:
            w = F.softplus(mod.weight) # Ensure positive for sinkhorn logic
            
            # Penalize deviation from mean (variance of sums)
            r_var = torch.var(w.sum(dim=1))
            c_var = torch.var(w.sum(dim=0))
            
            loss += r_var + c_var
            
        return self.weight_decay * loss
