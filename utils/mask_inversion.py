"""
Mask Inversion for localized feature extraction.

This module implements the Mask Inversion technique for computing
embeddings that are localized to specific regions of interest in images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import math
import types


def min_max(x: torch.Tensor, eps: float = 1e-6):
    """Min-max normalization."""
    return (x - x.amin(dim=(-1, -2, -3, -4), keepdim=True)) / (
        x.amax(dim=(-1, -2, -3, -4), keepdim=True)
        - x.amin(dim=(-1, -2, -3, -4), keepdim=True)
        + eps
    )


def hooked_torch_multi_head_attention_forward_dino_B(
    self, hidden_states, head_mask=None, output_attentions=False
):
    """
    Hooked attention forward that stores attention maps with gradients enabled.
    
    This is a critical function that ensures attention_maps retain gradients
    for the LeGrad explainability computation, even when LoRA adapters are used.
    
    Note: This hooks the Dinov2Attention module (self = layer.attention)
    The self.attention is actually Dinov2SelfAttention which has query/key/value.
    """
    # QKV projections - self.attention is Dinov2SelfAttention
    mixed_query_layer = self.attention.query(hidden_states)
    key_layer = self.attention.key(hidden_states)
    value_layer = self.attention.value(hidden_states)

    # Reshape and transpose
    batch_size, seq_length, hidden_size = hidden_states.size()
    num_heads = self.attention.num_attention_heads
    head_dim = self.attention.attention_head_size

    key_layer = key_layer.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    value_layer = value_layer.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)
    query_layer = mixed_query_layer.view(batch_size, seq_length, num_heads, head_dim).transpose(1, 2)

    # Attention computation - explicitly require gradients
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    attention_scores = attention_scores * self.attention.scaling

    # Ensure attention_probs requires grad for LeGrad computation
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    
    # Force requires_grad for explainability gradient computation
    if not attention_probs.requires_grad and torch.is_grad_enabled():
        attention_probs = attention_probs.detach().requires_grad_(True)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    # Store attention maps for LeGrad
    self.attention_maps = attention_probs

    # Output computation
    context_layer = torch.matmul(attention_probs, value_layer)
    context_layer = context_layer.transpose(1, 2).contiguous()
    context_layer = context_layer.view(batch_size, seq_length, hidden_size)

    context_layer = self.output.dense(context_layer)
    context_layer = self.output.dropout(context_layer)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    return outputs


def hooked_resblock_forward_dino_B(self, hidden_states, head_mask=None, output_attentions=False):
    """
    Hooked residual block forward that stores intermediate features.
    
    Note: This hooks the Dinov2Layer module.
    Uses norm1/norm2 instead of layernorm_before/layernorm_after for HuggingFace.
    """
    self_attention_outputs = self.attention(
        self.norm1(hidden_states),
        head_mask=head_mask,
        output_attentions=output_attentions,
    )
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # attn weights if output_attentions

    # Apply layer_scale1 (it's a module, needs to be called)
    if hasattr(self, 'layer_scale1') and self.layer_scale1 is not None:
        attention_output = self.layer_scale1(attention_output)
    
    # Apply drop_path
    if hasattr(self, 'drop_path') and self.drop_path is not None:
        attention_output = self.drop_path(attention_output)
    
    # First residual connection
    hidden_states = attention_output + hidden_states

    # MLP with layer norm
    mlp_output = self.mlp(self.norm2(hidden_states))
    
    # Apply layer_scale2 (it's a module, needs to be called)
    if hasattr(self, 'layer_scale2') and self.layer_scale2 is not None:
        mlp_output = self.layer_scale2(mlp_output)
    
    # Apply drop_path
    if hasattr(self, 'drop_path') and self.drop_path is not None:
        mlp_output = self.drop_path(mlp_output)
    
    # Second residual connection
    hidden_states = hidden_states + mlp_output

    # Store for LeGrad
    self.feat_post_mlp = hidden_states

    if output_attentions:
        outputs = (hidden_states,) + outputs
    else:
        outputs = (hidden_states,)

    return outputs


# ============== SigLIP Hooked Functions ==============

def hooked_siglip_attention_forward(
    self, hidden_states, attention_mask=None, output_attentions=False
):
    """
    Hooked attention forward for SigLIP that stores attention maps with gradients.
    
    SigLIP uses q_proj, k_proj, v_proj instead of query, key, value.
    """
    batch_size, seq_length, embed_dim = hidden_states.size()
    
    # QKV projections
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    # Reshape to multi-head
    query_states = query_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
    
    # Attention computation
    attention_scores = torch.matmul(query_states, key_states.transpose(-1, -2)) * self.scale
    
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    
    # Softmax
    attention_probs = nn.functional.softmax(attention_scores, dim=-1)
    
    # Force requires_grad for explainability gradient computation
    if not attention_probs.requires_grad and torch.is_grad_enabled():
        attention_probs = attention_probs.detach().requires_grad_(True)
    
    # Store attention maps for LeGrad
    self.attention_maps = attention_probs
    
    # Apply dropout (dropout is a float in SigLIP, use F.dropout)
    if self.training and self.dropout > 0:
        attention_probs_dropped = F.dropout(attention_probs, p=self.dropout, training=self.training)
    else:
        attention_probs_dropped = attention_probs
    
    # Compute output
    attn_output = torch.matmul(attention_probs_dropped, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_length, embed_dim)
    attn_output = self.out_proj(attn_output)
    
    return attn_output, attention_probs if output_attentions else None


def hooked_siglip_encoder_layer_forward(
    self, hidden_states, attention_mask=None, output_attentions=False
):
    """
    Hooked encoder layer forward for SigLIP that stores intermediate features.
    """
    # Self attention with pre-norm
    residual = hidden_states
    hidden_states = self.layer_norm1(hidden_states)
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = residual + hidden_states
    
    # MLP with pre-norm
    residual = hidden_states
    hidden_states = self.layer_norm2(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    
    # Store for LeGrad
    self.feat_post_mlp = hidden_states
    
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    
    return outputs


def batched_dice_loss(input, target, smooth=1e-6):
    """
    Compute batched Dice loss between input and target masks.
    
    Args:
        input: Predicted masks [B, C, H, W]
        target: Target masks [B, C, H, W]
        smooth: Smoothing factor
        
    Returns:
        Dice loss value
    """
    input = input.view(input.size(0), -1)
    target = target.view(target.size(0), -1)
    
    intersection = (input * target).sum(dim=1)
    union = input.sum(dim=1) + target.sum(dim=1)
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()


class LeWrapper(torch.nn.Module):
    """
    Wrapper for computing explainability maps using gradient-based methods.
    
    This implementation matches the original LeGrad wrapper behavior,
    especially for HuggingFace DINOv2-B models with LoRA adapters.
    """
    
    def __init__(self, model, layer_index=-1):
        super().__init__()
        self.model = model
        self.layer_index = layer_index
        self.visual = model
        self.args = getattr(model, 'args', None)
        self.model_type = "dinov2_B"
        self.patch_size = 14  # DINOv2 default patch size
        
        # Get starting depth
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            num_layers = len(model.encoder.layer)
        elif hasattr(model, 'blocks'):
            num_layers = len(model.blocks)
        else:
            num_layers = 12  # Default
        
        self.starting_depth = layer_index if layer_index >= 0 else num_layers + layer_index
        
        # Activate hooks
        self._activate_hooks()
        
    def _activate_hooks(self):
        """Activate self-attention hooks for gradient computation."""
        # Disable gradients for all params first
        for name, param in self.named_parameters():
            param.requires_grad = False
            
            # Enable gradients for encoder layers at or after starting_depth
            if "encoder.layer" in name or "model.encoder.layer" in name:
                # Extract depth from parameter name
                parts = name.split("encoder.layer")
                if len(parts) > 1:
                    depth_str = parts[-1].split(".")[0].replace(".", "")
                    try:
                        depth = int(depth_str)
                        if depth >= self.starting_depth:
                            param.requires_grad = True
                    except ValueError:
                        pass
        
        # Apply hooked forward methods
        if hasattr(self.visual, 'encoder') and hasattr(self.visual.encoder, 'layer'):
            for layer_idx in range(self.starting_depth, len(self.visual.encoder.layer)):
                layer = self.visual.encoder.layer[layer_idx]
                layer.layer_num = layer_idx
                
                # Hook the attention module
                layer.attention.forward = types.MethodType(
                    hooked_torch_multi_head_attention_forward_dino_B,
                    layer.attention
                )
                
                # Hook the layer forward
                layer.forward = types.MethodType(
                    hooked_resblock_forward_dino_B,
                    layer
                )
    
    def encode_image(self, image):
        """Encode image through the visual model."""
        if hasattr(self.model, 'forward'):
            if isinstance(image, dict):
                outputs = self.model(**image)
            else:
                outputs = self.model(pixel_values=image)
            
            if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                return outputs.last_hidden_state[:, 0]
            else:
                return outputs
        return self.model(image)
    
    def compute_legrad(self, text_embedding, image=None):
        """
        Compute gradient-based explainability map.
        
        Args:
            text_embedding: Query embedding [P, D]
            image: Optional image tensor to encode first
            
        Returns:
            Explainability map [1, P, H, W]
        """
        return self.compute_legrad_dinov2_B(text_embedding, image)
    
    def compute_legrad_dinov2_B(self, text_embedding, image=None):
        """
        Compute LeGrad explainability map for DINOv2-B models.
        
        This method replicates the original's behavior with proper gradient
        computation through attention maps.
        """
        from einops import rearrange
        
        num_prompts = text_embedding.shape[0]
        
        if image is not None:
            image = image.repeat(num_prompts, 1, 1, 1)
            _ = self.encode_image(image)
        
        # Get blocks list
        blocks_list = list(
            dict(self.visual.encoder.layer.named_children()).values()
        )
        
        image_features_list = []
        
        # Collect intermediate features from each layer
        for layer_idx in range(self.starting_depth, len(self.visual.encoder.layer)):
            layer = self.visual.encoder.layer[layer_idx]
            
            if hasattr(layer, 'feat_post_mlp'):
                intermediate_feat = layer.feat_post_mlp  # [batch, num_patch, dim]
                # Take mean over patches and apply layer norm
                intermediate_feat = self.visual.layernorm(
                    intermediate_feat.mean(dim=1)
                )
                intermediate_feat = F.normalize(intermediate_feat, dim=-1)
                image_features_list.append(intermediate_feat)
        
        if not image_features_list:
            # Fallback: return None if no features collected
            return None
        
        # Get spatial dimensions from last layer
        last_layer = blocks_list[-1]
        if hasattr(last_layer, 'feat_post_mlp'):
            num_tokens = last_layer.feat_post_mlp.shape[1] - 1
            w = h = int(math.sqrt(num_tokens))
        else:
            # Default to 16x16 for 224x224 images with patch_size=14
            w = h = 16
        
        # Compute explainability map
        accum_expl_map = 0
        
        for layer_offset, (blk, img_feat) in enumerate(
            zip(blocks_list[self.starting_depth:], image_features_list)
        ):
            self.visual.zero_grad()
            
            # Compute similarity
            sim = text_embedding @ img_feat.transpose(-1, -2)  # [P, 1]
            
            one_hot = (
                F.one_hot(torch.arange(0, num_prompts))
                .float()
                .requires_grad_(True)
                .to(text_embedding.device)
            )
            one_hot = torch.sum(one_hot * sim)
            
            # Get attention map from the block
            attn_map = blocks_list[
                self.starting_depth + layer_offset
            ].attention.attention_maps  # [b, num_heads, N, N]
            
            if attn_map is None:
                continue
            
            # Compute gradient
            try:
                grad = torch.autograd.grad(
                    one_hot, [attn_map], retain_graph=True, create_graph=True
                )[0]
            except RuntimeError:
                # If gradient computation fails, skip this layer
                continue
            
            # Reshape if needed
            if grad.ndim == 3:
                grad = rearrange(
                    grad, "(b h) n m -> b h n m", b=num_prompts
                )
            
            grad = torch.clamp(grad, min=0.0)
            
            # Compute image relevance
            image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # Skip CLS token
            
            # Reshape to [B, 1, H, W] for Dice loss compatibility
            expl_map = rearrange(
                image_relevance, "b (w h) -> b 1 w h", w=w, h=h
            )
            expl_map = F.interpolate(
                expl_map, scale_factor=self.patch_size, mode="bilinear"
            )
            accum_expl_map = accum_expl_map + expl_map
        
        # Min-max normalization
        if isinstance(accum_expl_map, torch.Tensor):
            min_val = accum_expl_map.min()
            max_val = accum_expl_map.max()
            if max_val - min_val > 1e-6:
                accum_expl_map = (accum_expl_map - min_val) / (max_val - min_val)
        
        return accum_expl_map
    
    def forward(self, x=None, **kwargs):
        """Forward pass through the wrapped model."""
        if x is not None:
            return self.model(x)
        elif 'pixel_values' in kwargs:
            return self.model(**kwargs)
        else:
            return self.model(**kwargs)
    
    def zero_grad(self):
        self.model.zero_grad()


class MaskInversion(LeWrapper):
    """
    Mask Inversion module for computing localized embeddings.
    
    This technique optimizes an embedding to match a target mask region
    while staying close to the original image embedding.
    """
    
    def __init__(
        self,
        model,
        layer_index=-1,
        alpha=0.0,
        beta=0.0,
        lr=0.1,
        iterations=100,
        wd=0.0,
        optimizer=optim.AdamW
    ):
        """
        Initialize Mask Inversion.
        
        Args:
            model: Feature extraction model
            layer_index: Layer to compute gradients from
            alpha: Regularization weight for staying close to original embedding
            beta: Orthogonality regularization weight
            lr: Learning rate for optimization
            iterations: Number of optimization iterations
            wd: Weight decay
            optimizer: Optimizer class to use
        """
        super().__init__(model, layer_index)
        self.float()
        self.optimizer_class = optimizer
        self.iterations = iterations
        self.lr = lr
        self.wd = wd
        self.alpha = alpha
        self.beta = beta
    
    def _get_image_features(self, image):
        """Extract features from image tensor."""
        if hasattr(self.model, 'encode_image'):
            return self.model.encode_image(image)
        elif hasattr(self.model, 'forward'):
            # For HuggingFace models that expect pixel_values
            if hasattr(self.model, 'config'):
                outputs = self.model(pixel_values=image)
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    return outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # Use CLS token
                    return outputs.last_hidden_state[:, 0]
                else:
                    return outputs
            else:
                return self.model(image)
        else:
            return self.visual(image)
    
    def compute_maskinversion(
        self,
        image,
        masks_target,
        alpha=None,
        lr=None,
        iterations=None,
        wd=None,
        verbose=False,
        return_expl_map=False
    ):
        """
        Compute mask-inverted embeddings.
        
        Args:
            image: Input image tensor [B, C, H, W] or dict with 'x' key
            masks_target: Target masks [N, H, W]
            alpha: Override regularization weight
            lr: Override learning rate
            iterations: Override number of iterations
            wd: Override weight decay
            verbose: Print progress
            return_expl_map: Return explainability maps
            
        Returns:
            Optimized embeddings, optionally with explainability maps
        """
        # Update hyperparameters if provided
        if alpha is not None:
            self.alpha = alpha
        if lr is not None:
            self.lr = lr
        if iterations is not None:
            self.iterations = iterations
        if wd is not None:
            self.wd = wd
        
        # Handle dict input
        if isinstance(image, dict):
            key = list(image.keys())[0]
            image_tensor = image[key]
        else:
            image_tensor = image
        
        # Get original image features
        image_features = self._get_image_features(image_tensor)
        if isinstance(image_features, dict):
            image_features = image_features.get('pooler_output', 
                                                list(image_features.values())[0])
        
        # Initialize mask embeddings
        mask_emb = image_features.detach().clone()
        if masks_target.shape[0] > 1:
            mask_emb = mask_emb.expand(masks_target.shape[0], -1)
        mask_emb = mask_emb.requires_grad_(True)
        
        # Initialize optimizer
        optimizer = self.optimizer_class(
            [mask_emb], 
            lr=self.lr, 
            weight_decay=self.wd
        )
        
        start = time.time()
        first_expl_map = None
        expl_map = None
        
        for it in range(self.iterations):
            self.zero_grad()
            
            # Normalize embedding
            mask_emb_norm = F.normalize(mask_emb, dim=-1)
            
            # Compute explainability map
            expl_map = self.compute_legrad(mask_emb_norm)
            if expl_map is not None:
                expl_map = F.interpolate(
                    expl_map, 
                    size=image_tensor.shape[-2:], 
                    mode="bilinear",
                    align_corners=False
                )
                
                if it == 0:
                    first_expl_map = expl_map.detach().clone()
            
            # Compute losses
            loss_reg = (
                1 - (F.normalize(image_features, dim=-1) * mask_emb_norm).sum(dim=-1)
            ).mean()
            
            if expl_map is not None:
                loss_mask = batched_dice_loss(expl_map, masks_target.unsqueeze(1))
            else:
                loss_mask = torch.tensor(0.0, device=mask_emb.device)
            
            # Orthogonality loss for multiple masks
            if mask_emb.shape[0] > 1:
                cos_sim = torch.mm(mask_emb_norm, mask_emb_norm.t())
                mask_d = torch.ones_like(cos_sim) - torch.eye(
                    cos_sim.size(0), device=cos_sim.device
                )
                loss_orth = (cos_sim.abs() * mask_d).sum() / mask_d.sum()
            else:
                loss_orth = 0
            
            loss = loss_mask + self.alpha * loss_reg + self.beta * loss_orth
            
            if verbose and it % 10 == 0:
                print(f"iteration {it}| loss: {loss.item():.4f}")
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        if verbose:
            print(f"total mask inversion time: {time.time() - start:.4f} s")
        
        if return_expl_map:
            return mask_emb, expl_map, first_expl_map
        return mask_emb
    
    def compute_maskinversion_sum(
        self,
        image,
        masks_target,
        image_sc=None,
        single_from_sc=False,
        alpha_sum_full=0,
        full_mask=None,
        alpha=None,
        lr=None,
        alpha_sum=0,
        iterations=None,
        wd=None,
        verbose=False,
        return_expl_map=False
    ):
        """
        Compute mask-inverted embeddings with sum loss.
        
        This variant optimizes a single embedding to match the sum of all masks,
        useful for gallery images with multiple objects.
        
        Args:
            image: Full image tensor
            masks_target: Target masks (from smart crop)
            image_sc: Smart-cropped images (optional)
            single_from_sc: Use single embedding from smart crop
            alpha_sum_full: Weight for full mask similarity
            full_mask: Full combined mask
            alpha: Regularization weight
            lr: Learning rate
            alpha_sum: Weight for smart crop similarity
            iterations: Number of iterations
            wd: Weight decay
            verbose: Print progress
            return_expl_map: Return explainability maps
            
        Returns:
            Optimized embeddings, optionally with explainability maps
        """
        if alpha is not None:
            self.alpha = alpha
        if lr is not None:
            self.lr = lr
        if iterations is not None:
            self.iterations = iterations
        if wd is not None:
            self.wd = wd
        
        sc_loss = alpha_sum > 0
        sc_loss_full = alpha_sum_full > 0
        
        if single_from_sc and image_sc is not None:
            # Compute embeddings from smart crops first
            mask_emb_sc, expl_map, first_expl_map = self.compute_maskinversion(
                image=image_sc,
                masks_target=masks_target,
                verbose=verbose,
                return_expl_map=True
            )
            return mask_emb_sc, expl_map, first_expl_map
        
        # Get pre-computed smart crop embeddings if needed
        mask_emb_sc = None
        if sc_loss or sc_loss_full:
            target_mask = full_mask if sc_loss_full else masks_target
            target_img = image if sc_loss_full else image_sc
            if target_img is not None:
                mask_emb_sc = self.compute_maskinversion(target_img, target_mask)
                mask_emb_sc_norm = F.normalize(mask_emb_sc, dim=-1)
        
        # Combine masks for sum loss
        if full_mask is not None:
            combined_mask = full_mask.sum(dim=0, keepdim=True)
            combined_mask[combined_mask > 0] = 1
            masks_target = combined_mask
        
        # Get image features
        image_features = self._get_image_features(image)
        if isinstance(image_features, dict):
            image_features = image_features.get('pooler_output',
                                                list(image_features.values())[0])
        
        # Initialize embedding (sum of smart crop embeddings if available)
        if mask_emb_sc is not None:
            mask_emb = mask_emb_sc.detach().clone().sum(dim=0, keepdim=True)
        else:
            mask_emb = image_features.detach().clone()
        mask_emb = mask_emb.requires_grad_(True)
        
        optimizer = self.optimizer_class(
            [mask_emb],
            lr=self.lr,
            weight_decay=self.wd
        )
        
        start = time.time()
        first_expl_map = None
        
        for it in range(self.iterations):
            self.zero_grad()
            
            mask_emb_norm = F.normalize(mask_emb, dim=-1)
            
            expl_map = self.compute_legrad(mask_emb_norm)
            if expl_map is not None:
                expl_map = F.interpolate(
                    expl_map,
                    size=image.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                
                if it == 0:
                    first_expl_map = expl_map.detach().clone()
            
            # Losses
            loss_sim_sc = 0
            if mask_emb_sc is not None:
                loss_sim_sc = (1 - torch.mm(mask_emb_norm, mask_emb_sc_norm.t())[0]).mean()
            
            loss_reg = (
                1 - (F.normalize(image_features, dim=-1) * mask_emb_norm).sum(dim=-1)
            ).mean()
            
            if expl_map is not None:
                loss_mask = batched_dice_loss(expl_map, masks_target)
            else:
                loss_mask = torch.tensor(0.0, device=mask_emb.device)
            
            alpha_sum_weight = max(alpha_sum, alpha_sum_full)
            loss = loss_mask + self.alpha * loss_reg + alpha_sum_weight * loss_sim_sc
            
            if verbose and it % 10 == 0:
                print(f"iteration {it}| loss: {loss.item():.4f}")
            
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        if verbose:
            print(f"total mask inversion time: {time.time() - start:.4f} s")
        
        if return_expl_map:
            return mask_emb, expl_map, first_expl_map
        return mask_emb
    
    def extract_mask_from_path(self, img_path, img_pil):
        """
        Extract masks from captions file for a given image path.
        
        Args:
            img_path: Path to the image
            img_pil: PIL Image
            
        Returns:
            List of masks as numpy arrays
        """
        if not hasattr(self, 'captions') or self.captions is None:
            # Return full image mask if no captions
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        if img_path not in self.captions:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        masks = self.captions[img_path].get('masks', [])
        if len(masks) == 0:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        return masks


class MaskInversionSigLIP(MaskInversion):
    """
    Mask Inversion module specialized for SigLIP models.
    
    SigLIP has a different architecture from DINOv2:
    - Uses SiglipEncoderLayer with layer_norm1, layer_norm2
    - Attention uses q_proj, k_proj, v_proj instead of query, key, value
    - Has a different output structure with head/pooler
    """
    
    def __init__(
        self,
        model,
        layer_index=-1,
        alpha=0.0,
        beta=0.0,
        lr=0.1,
        iterations=100,
        wd=0.0,
        optimizer=optim.AdamW
    ):
        # Don't call parent __init__ directly to avoid DINOv2 hook activation
        torch.nn.Module.__init__(self)
        
        # Use object.__setattr__ to avoid nn.Module's attribute interception
        # This ensures these attributes are set in __dict__ directly
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'layer_index', layer_index)
        visual = model.vision_model if hasattr(model, 'vision_model') else model
        object.__setattr__(self, 'visual', visual)
        object.__setattr__(self, 'args', getattr(model, 'args', None))
        object.__setattr__(self, 'model_type', "siglip")
        object.__setattr__(self, 'patch_size', 16)  # SigLIP base uses 16x16 patches
        
        # Get number of layers
        if hasattr(visual, 'encoder') and hasattr(visual.encoder, 'layers'):
            num_layers = len(visual.encoder.layers)
        else:
            num_layers = 12
        
        starting_depth = layer_index if layer_index >= 0 else num_layers + layer_index
        object.__setattr__(self, 'starting_depth', starting_depth)
        
        # Initialize optimization parameters
        self.float()
        object.__setattr__(self, 'optimizer_class', optimizer)
        object.__setattr__(self, 'iterations', iterations)
        object.__setattr__(self, 'lr', lr)
        object.__setattr__(self, 'wd', wd)
        object.__setattr__(self, 'alpha', alpha)
        object.__setattr__(self, 'beta', beta)
        
        # Activate hooks
        self._activate_hooks_siglip()
    
    def __getattr__(self, name):
        """Proxy attribute access to the underlying model for compatibility."""
        # For nn.Module subclasses, certain attributes might be stored differently
        # Check _modules, _parameters, _buffers first (standard nn.Module storage)
        # Then check the object's __dict__
        
        # If 'model' is not in our dict, we can't proxy
        obj_dict = object.__getattribute__(self, '__dict__')
        if 'model' not in obj_dict:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        model = obj_dict['model']
        if hasattr(model, name):
            return getattr(model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def _activate_hooks_siglip(self):
        """Activate self-attention hooks for SigLIP gradient computation."""
        # Disable gradients for all params first
        for name, param in self.named_parameters():
            param.requires_grad = False
            
            # Enable gradients for encoder layers at or after starting_depth
            if "encoder.layers" in name:
                parts = name.split("encoder.layers.")
                if len(parts) > 1:
                    depth_str = parts[-1].split(".")[0]
                    try:
                        depth = int(depth_str)
                        if depth >= self.starting_depth:
                            param.requires_grad = True
                    except ValueError:
                        pass
        
        # Apply hooked forward methods
        if hasattr(self.visual, 'encoder') and hasattr(self.visual.encoder, 'layers'):
            for layer_idx in range(self.starting_depth, len(self.visual.encoder.layers)):
                layer = self.visual.encoder.layers[layer_idx]
                layer.layer_num = layer_idx
                
                # Hook the attention module
                layer.self_attn.forward = types.MethodType(
                    hooked_siglip_attention_forward,
                    layer.self_attn
                )
                
                # Hook the layer forward
                layer.forward = types.MethodType(
                    hooked_siglip_encoder_layer_forward,
                    layer
                )
    
    def encode_image(self, image):
        """Encode image through SigLIP vision model."""
        if isinstance(image, dict):
            pixel_values = image.get('pixel_values', image.get('x', list(image.values())[0]))
        else:
            pixel_values = image
        
        outputs = self.visual(pixel_values=pixel_values)
        
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state.mean(dim=1)
        else:
            return outputs
    
    def compute_legrad(self, text_embedding, image=None):
        """Compute LeGrad explainability map for SigLIP."""
        return self.compute_legrad_siglip(text_embedding, image)
    
    def compute_legrad_siglip(self, text_embedding, image=None):
        """
        Compute LeGrad explainability map for SigLIP models.
        """
        from einops import rearrange
        
        num_prompts = text_embedding.shape[0]
        
        if image is not None:
            image = image.repeat(num_prompts, 1, 1, 1)
            _ = self.encode_image(image)
        
        # Get blocks list
        blocks_list = list(self.visual.encoder.layers)
        
        image_features_list = []
        
        # Collect intermediate features from each layer
        for layer_idx in range(self.starting_depth, len(blocks_list)):
            layer = blocks_list[layer_idx]
            
            if hasattr(layer, 'feat_post_mlp'):
                intermediate_feat = layer.feat_post_mlp  # [batch, num_patch, dim]
                # Apply post layer norm and mean pooling
                if hasattr(self.visual, 'post_layernorm'):
                    intermediate_feat = self.visual.post_layernorm(intermediate_feat)
                intermediate_feat = intermediate_feat.mean(dim=1)
                intermediate_feat = F.normalize(intermediate_feat, dim=-1)
                image_features_list.append(intermediate_feat)
        
        if not image_features_list:
            return None
        
        # Get spatial dimensions from last layer
        last_layer = blocks_list[-1]
        if hasattr(last_layer, 'feat_post_mlp'):
            num_tokens = last_layer.feat_post_mlp.shape[1]
            w = h = int(math.sqrt(num_tokens))
        else:
            # Default for 384x384 images with 16x16 patches
            w = h = 24
        
        # Compute explainability map
        accum_expl_map = 0
        
        for layer_offset, (blk, img_feat) in enumerate(
            zip(blocks_list[self.starting_depth:], image_features_list)
        ):
            self.visual.zero_grad()
            
            # Compute similarity
            sim = text_embedding @ img_feat.transpose(-1, -2)
            
            one_hot = (
                F.one_hot(torch.arange(0, num_prompts))
                .float()
                .requires_grad_(True)
                .to(text_embedding.device)
            )
            one_hot = torch.sum(one_hot * sim)
            
            # Get attention map from the block
            attn_map = blocks_list[
                self.starting_depth + layer_offset
            ].self_attn.attention_maps
            
            if attn_map is None:
                continue
            
            # Compute gradient
            try:
                grad = torch.autograd.grad(
                    one_hot, [attn_map], retain_graph=True, create_graph=True
                )[0]
            except RuntimeError:
                continue
            
            # Reshape if needed
            if grad.ndim == 3:
                grad = rearrange(
                    grad, "(b h) n m -> b h n m", b=num_prompts
                )
            
            grad = torch.clamp(grad, min=0.0)
            
            # Compute image relevance (SigLIP doesn't have CLS token)
            image_relevance = grad.mean(dim=1).mean(dim=1)
            
            # Reshape to [B, 1, H, W]
            expl_map = rearrange(
                image_relevance, "b (w h) -> b 1 w h", w=w, h=h
            )
            expl_map = F.interpolate(
                expl_map, scale_factor=self.patch_size, mode="bilinear"
            )
            accum_expl_map = accum_expl_map + expl_map
        
        # Min-max normalization
        if isinstance(accum_expl_map, torch.Tensor):
            min_val = accum_expl_map.min()
            max_val = accum_expl_map.max()
            if max_val - min_val > 1e-6:
                accum_expl_map = (accum_expl_map - min_val) / (max_val - min_val)
        
        return accum_expl_map
    
    def _get_image_features(self, image):
        """Extract features from image tensor for SigLIP."""
        return self.encode_image(image)


# =============================================================================
# CLIP Hooked Attention Functions  
# =============================================================================

def hooked_clip_multi_head_attention_forward(
    self,
    query,
    key,
    value,
    key_padding_mask=None,
    need_weights=True,
    attn_mask=None,
):
    """
    Hooked attention forward for OpenAI CLIP's MultiheadAttention.
    
    This stores attention maps for gradient-based explainability.
    """
    # Standard multi-head attention forward
    L, N, E = query.shape
    
    # Compute Q, K, V
    if self._qkv_same_embed_dim:
        q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
    else:
        q = F.linear(query, self.q_proj_weight, self.in_proj_bias[:E])
        k = F.linear(key, self.k_proj_weight, self.in_proj_bias[E:2*E])
        v = F.linear(value, self.v_proj_weight, self.in_proj_bias[2*E:])
    
    # Reshape for multi-head attention
    head_dim = E // self.num_heads
    q = q.contiguous().view(L, N * self.num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(L, N * self.num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(L, N * self.num_heads, head_dim).transpose(0, 1)
    
    # Compute attention scores
    scale = float(head_dim) ** -0.5
    attn_scores = torch.bmm(q * scale, k.transpose(-1, -2))
    
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask
        attn_scores += attn_mask
    
    # Softmax
    attn_probs = F.softmax(attn_scores, dim=-1)
    
    # Ensure attention_probs requires grad for LeGrad computation
    if not attn_probs.requires_grad and torch.is_grad_enabled():
        attn_probs = attn_probs.detach().requires_grad_(True)
    
    # Store attention maps for explainability
    self.attention_maps = attn_probs
    
    # Dropout (during training)
    if self.training and self.dropout > 0:
        attn_probs = F.dropout(attn_probs, p=self.dropout)
    
    # Compute output
    attn_output = torch.bmm(attn_probs, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(L, N, E)
    attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
    
    return attn_output, self.attention_maps


def hooked_clip_resblock_forward(
    self,
    q_x: torch.Tensor,
    k_x=None,
    v_x=None,
    attn_mask=None,
):
    """
    Hooked ResidualAttentionBlock forward for CLIP.
    
    Stores intermediate features for explainability.
    """
    # Layer norm before attention
    x_norm = self.ln_1(q_x)
    
    # Self-attention
    attn_output, attn_weights = self.attn(
        x_norm, x_norm, x_norm,
        need_weights=True,
        attn_mask=attn_mask
    )
    
    # Residual connection
    x = q_x + attn_output
    self.feat_post_attn = x
    
    # MLP with layer norm
    x = x + self.mlp(self.ln_2(x))
    self.feat_post_mlp = x
    
    return x


class MaskInversionCLIP(MaskInversion):
    """
    Mask Inversion module specialized for OpenAI CLIP models.
    
    CLIP uses a VisionTransformer with:
    - transformer.resblocks: list of ResidualAttentionBlock
    - Each block has: attn (MultiheadAttention), ln_1, mlp, ln_2
    """
    
    def __init__(
        self,
        model,
        layer_index=-1,
        alpha=0.0,
        beta=0.0,
        lr=0.1,
        iterations=100,
        wd=0.0,
        optimizer=optim.AdamW
    ):
        # Don't call parent __init__ directly to avoid DINOv2 hook activation
        torch.nn.Module.__init__(self)
        
        # Use object.__setattr__ to avoid nn.Module's attribute interception
        object.__setattr__(self, 'model', model)
        object.__setattr__(self, 'layer_index', layer_index)
        
        # For CLIP, the visual model is model.visual
        visual = model.visual if hasattr(model, 'visual') else model
        object.__setattr__(self, 'visual', visual)
        object.__setattr__(self, 'args', getattr(model, 'args', None))
        object.__setattr__(self, 'model_type', "clip")
        
        # Get patch size from conv1
        patch_size = visual.conv1.kernel_size[0] if hasattr(visual, 'conv1') else 16
        object.__setattr__(self, 'patch_size', patch_size)
        
        # Get number of layers
        if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
            num_layers = len(visual.transformer.resblocks)
        else:
            num_layers = 12
        
        starting_depth = layer_index if layer_index >= 0 else num_layers + layer_index
        object.__setattr__(self, 'starting_depth', starting_depth)
        
        # Initialize optimization parameters
        self.float()
        object.__setattr__(self, 'optimizer_class', optimizer)
        object.__setattr__(self, 'iterations', iterations)
        object.__setattr__(self, 'lr', lr)
        object.__setattr__(self, 'wd', wd)
        object.__setattr__(self, 'alpha', alpha)
        object.__setattr__(self, 'beta', beta)
        
        # Activate hooks
        self._activate_hooks_clip()
    
    def __getattr__(self, name):
        """Proxy attribute access to the underlying model for compatibility."""
        obj_dict = object.__getattribute__(self, '__dict__')
        if 'model' not in obj_dict:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        model = obj_dict['model']
        if hasattr(model, name):
            return getattr(model, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def _activate_hooks_clip(self):
        """Activate self-attention hooks for CLIP gradient computation."""
        # Disable gradients for all params first
        for name, param in self.named_parameters():
            param.requires_grad = False
            
            # Enable gradients for encoder layers at or after starting_depth
            if "transformer.resblocks" in name or "visual.transformer.resblocks" in name:
                # Extract depth from name
                if "visual.transformer.resblocks" in name:
                    depth_str = name.split("visual.transformer.resblocks.")[-1].split(".")[0]
                else:
                    depth_str = name.split("transformer.resblocks.")[-1].split(".")[0]
                try:
                    depth = int(depth_str)
                    if depth >= self.starting_depth:
                        param.requires_grad = True
                except ValueError:
                    pass
        
        # Apply hooked forward methods
        if hasattr(self.visual, 'transformer') and hasattr(self.visual.transformer, 'resblocks'):
            for layer_idx in range(self.starting_depth, len(self.visual.transformer.resblocks)):
                layer = self.visual.transformer.resblocks[layer_idx]
                layer.layer_num = layer_idx
                
                # Hook the attention module
                layer.attn.forward = types.MethodType(
                    hooked_clip_multi_head_attention_forward,
                    layer.attn
                )
                
                # Hook the layer forward
                layer.forward = types.MethodType(
                    hooked_clip_resblock_forward,
                    layer
                )
    
    def encode_image(self, image):
        """Encode image through CLIP vision model."""
        if isinstance(image, dict):
            pixel_values = image.get('pixel_values', image.get('x', list(image.values())[0]))
        else:
            pixel_values = image
        
        # Use model's encode_image method
        return self.model.encode_image(pixel_values).float()
    
    def compute_legrad(self, text_embedding, image=None):
        """
        Compute gradient-based explainability map for CLIP.
        """
        return self.compute_legrad_clip(text_embedding, image)
    
    def compute_legrad_clip(self, text_embedding, image=None):
        """
        Compute LeGrad explainability map for CLIP models.
        """
        from einops import rearrange
        
        num_prompts = text_embedding.shape[0]
        
        if image is not None:
            image = image.repeat(num_prompts, 1, 1, 1)
            _ = self.encode_image(image)
        
        # Get blocks list
        blocks_list = list(self.visual.transformer.resblocks)
        
        image_features_list = []
        
        # Collect intermediate features from each layer
        for layer_idx in range(self.starting_depth, len(blocks_list)):
            layer = blocks_list[layer_idx]
            
            if hasattr(layer, 'feat_post_mlp'):
                intermediate_feat = layer.feat_post_mlp  # [L, N, D] format for CLIP (L=patches, N=batch, D=dim)
                # Average over patch dimension (dim=0) as in original implementation
                intermediate_feat = intermediate_feat.mean(dim=0)  # [N, D] = [batch, 768]
                # Apply ln_post
                if hasattr(self.visual, 'ln_post'):
                    intermediate_feat = self.visual.ln_post(intermediate_feat)
                # Apply projection (768 -> 512)
                if hasattr(self.visual, 'proj') and self.visual.proj is not None:
                    intermediate_feat = intermediate_feat @ self.visual.proj
                intermediate_feat = F.normalize(intermediate_feat, dim=-1)
                image_features_list.append(intermediate_feat)
        
        if not image_features_list:
            return None
        
        # Get spatial dimensions
        last_layer = blocks_list[-1]
        if hasattr(last_layer, 'feat_post_mlp'):
            feat = last_layer.feat_post_mlp
            if feat.dim() == 3:
                # For CLIP: [L, N, D] where L = num_patches + 1
                num_tokens = feat.shape[0] - 1  # Exclude CLS token
            else:
                num_tokens = 196
            w = h = int(math.sqrt(num_tokens))
        else:
            w = h = 14  # Default for 224x224 images with patch_size=16
        
        # Compute explainability map
        accum_expl_map = 0
        
        for layer_offset, (blk, img_feat) in enumerate(
            zip(blocks_list[self.starting_depth:], image_features_list)
        ):
            self.visual.zero_grad()
            
            # Compute similarity
            sim = text_embedding @ img_feat.transpose(-1, -2)  # [P, 1]
            
            one_hot = (
                F.one_hot(torch.arange(0, num_prompts))
                .float()
                .requires_grad_(True)
                .to(text_embedding.device)
            )
            one_hot = torch.sum(one_hot * sim)
            
            # Get attention map from the block
            attn_map = blk.attn.attention_maps  # [b*num_heads, N, N]
            
            if attn_map is None:
                continue
            
            # Compute gradient
            try:
                grad = torch.autograd.grad(
                    one_hot, [attn_map], retain_graph=True, create_graph=True
                )[0]
            except RuntimeError:
                continue
            
            # Reshape if needed
            if grad.ndim == 3:
                # [b*num_heads, N, N] -> [b, num_heads, N, N]
                num_heads = grad.shape[0] // num_prompts
                grad = rearrange(
                    grad, "(b h) n m -> b h n m", b=num_prompts, h=num_heads
                )
            
            grad = torch.clamp(grad, min=0.0)
            
            # Compute image relevance - skip CLS token (average over attn heads)
            image_relevance = grad.mean(dim=1).mean(dim=1)[:, 1:]  # [b, num_patches]
            
            # Reshape to spatial using rearrange like original
            expl_map = rearrange(
                image_relevance, "b (w h) -> 1 b w h", w=w, h=h
            )  # [1, B, H, W]
            
            # Resize using scale_factor like original
            expl_map = F.interpolate(
                expl_map,
                scale_factor=self.patch_size,
                mode='bilinear',
                align_corners=False
            )  # [1, B, H*scale, W*scale]
            
            accum_expl_map = accum_expl_map + expl_map
        
        # Min-max normalization using original min_max function
        if isinstance(accum_expl_map, torch.Tensor):
            # Match the original's behavior
            accum_expl_map = min_max(accum_expl_map)
            # Ensure 4D output: [B, 1, H, W] for F.interpolate compatibility
            if accum_expl_map.ndim == 3:
                accum_expl_map = accum_expl_map.unsqueeze(1)
        
        return accum_expl_map
    
    def _get_image_features(self, image):
        """Extract features from image tensor for CLIP."""
        return self.encode_image(image)