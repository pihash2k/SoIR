"""
CLIP feature extractor with Mask Inversion for localized features.
"""

import os
import sys

# Add the package directory to path for imports
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import copy

from extractors.clip_extractor import CLIPExtractor
from utils.mask_inversion import MaskInversionCLIP


class CLIPMIExtractor(CLIPExtractor):
    """
    CLIP with Mask Inversion for localized feature extraction.
    
    Combines CLIP's visual encoder with Mask Inversion to produce
    embeddings that are localized to specific regions of interest.
    """
    
    def __init__(self, args=None, device=None):
        super().__init__(args=args, device=device)
    
    def init_model(self):
        """Initialize CLIP model and wrap with MaskInversion."""
        # First initialize the base model
        super().init_model()
        
        # Store base model reference
        self.base_model = self.model
        
        # Wrap with MaskInversion
        lr = 0.1
        alpha = getattr(self.args, 'mi_alpha', 0.03)
        wd = 0.0
        iterations = getattr(self.args, 'mi_iterations', 100)
        layer_index = getattr(self.args, 'mi_layer_index', -1)
        
        self.model = MaskInversionCLIP(
            self.base_model,
            lr=lr,
            layer_index=layer_index,
            alpha=alpha,
            wd=wd,
            optimizer=torch.optim.AdamW,
            iterations=iterations,
        )
        self.model.args = self.args
        
        # Mask transform for resizing
        from torchvision import transforms
        self.mask_transform = transforms.Resize(
            (224, 224),  # CLIP uses 224x224
            interpolation=transforms.InterpolationMode.NEAREST
        )
    
    def extract(self, image, is_query=False, masks=None):
        """
        Extract mask-inverted features from an image.
        
        Args:
            image: PIL Image or dict with 'image' and 'img_path' keys
            is_query: Whether this is a query image
            masks: Optional pre-computed masks
            
        Returns:
            Dict with 'keypoints' tensor
        """
        self.is_current_query = is_query
        
        # For queries or when global features requested, use base extraction
        if is_query and not getattr(self.args, 'mask_input', False):
            return super().extract(image, is_query=is_query, masks=masks)
        
        if getattr(self.args, 'global_features', False):
            return self._extract_global_features_single(image, masks=masks)
        else:
            return super().extract(image, is_query=is_query, masks=masks)
    
    def _extract_global_features_single(self, image, masks=None):
        """
        Extract mask-inverted global features.
        
        Args:
            image: PIL Image or dict
            masks: Optional masks
            
        Returns:
            Dict with 'keypoints' tensor
        """
        # Handle dict input
        if isinstance(image, dict):
            img_path = image.get('img_path', None)
            img_pil = image['image']
        else:
            img_path = None
            img_pil = image
        
        # Get masks
        if masks is None and img_path is not None:
            masks = self.extract_mask_from_path(img_path, img_pil)
        elif masks is None:
            # Full image mask
            masks = [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        # Resize image and masks
        img_pil, masks = self.resize_image_and_masks(img_pil, masks)
        
        # Handle full_mask option (combine all masks)
        if getattr(self.args, 'full_mask', False):
            combined = np.array(masks).sum(axis=0)
            combined[combined > 0] = 255
            masks = [combined.astype(np.uint8)]
        
        # Handle mask_input option (mask the image directly)
        if getattr(self.args, 'mask_input', False) and self.is_current_query:
            f_masks = np.array(masks).sum(axis=0)
            masked_im = np.array(img_pil) * (f_masks[:, :, np.newaxis] / 255)
            img_pil = Image.fromarray(masked_im.astype('uint8'))
        
        # Preprocess full image
        full_img = self._pre_process_input(img_pil)
        
        # If not smart crop, stack copies for each mask
        if not getattr(self.args, 'smart_crop', False):
            masks_all = copy.deepcopy(masks)
            masks_all = torch.from_numpy(np.array(masks_all).astype('int32'))
            masks_all = self._pre_process_mask(masks_all)
            masks_all = masks_all.to(self.device)
            masks_all = masks_all / 255.0
            masks_all = masks_all.float()
        
        # Smart crop: create individual crops for each mask
        if getattr(self.args, 'smart_crop', False):
            crops = self.crop_image_and_masks(I=img_pil, masks=masks)
            if len(crops) > 0:
                img_pils, masks = self.process_tuple_list(crops)
                # Resize cropped images and masks to standard size (224 for CLIP)
                resized_img_pils = []
                resized_masks = []
                for im, m in zip(img_pils, masks):
                    im_resized, m_list = self.resize_image_and_masks(im, [m], target_size=224)
                    resized_img_pils.append(im_resized)
                    resized_masks.append(m_list[0])
                img_pils = resized_img_pils
                masks = resized_masks
            else:
                img_pils = [img_pil]
        else:
            img_pils = [img_pil] * len(masks)
        
        # Preprocess all images
        images = [self._pre_process_input(im) for im in img_pils]
        
        # Stack into batch
        if isinstance(images[0], dict):
            key = list(images[0].keys())[0]
            image_batch = {'x': torch.cat([im[key] for im in images])}
        else:
            image_batch = torch.cat(images)
        
        # Preprocess masks
        masks_tensor = torch.from_numpy(np.array(masks).astype('int64'))
        masks_tensor = self._pre_process_mask(masks_tensor)
        masks_tensor = masks_tensor.to(self.device) / 255.0
        masks_tensor = masks_tensor.float()
        
        # Apply mask inversion
        if getattr(self.args, 'mi_sum', False) and not self.is_current_query:
            localized_embeddings, _, _ = self.model.compute_maskinversion_sum(
                image=full_img,
                image_sc=image_batch if getattr(self.args, 'smart_crop', False) else None,
                masks_target=masks_tensor,
                alpha=getattr(self.args, 'mi_alpha', 0.03),
                single_from_sc=getattr(self.args, 'single_from_sc', False),
                alpha_sum=0,
                alpha_sum_full=0,
                return_expl_map=True
            )
        else:
            localized_embeddings, _, _ = self.model.compute_maskinversion(
                image=image_batch,
                masks_target=masks_tensor,
                alpha=getattr(self.args, 'mi_alpha', 0.03),
                return_expl_map=True
            )
        
        # Normalize
        localized_embeddings = F.normalize(localized_embeddings, p=2, dim=-1)
        
        return {"keypoints": localized_embeddings}
    
    def resize_image_and_masks(self, img_pil, masks, target_size=224):
        """Resize image and masks to target size."""
        # Resize image
        img_resized = img_pil.resize((target_size, target_size), Image.Resampling.BILINEAR)
        
        # Resize masks
        masks_resized = []
        for mask in masks:
            if isinstance(mask, np.ndarray):
                mask_pil = Image.fromarray(mask.astype(np.uint8))
            else:
                mask_pil = mask
            mask_resized = mask_pil.resize((target_size, target_size), Image.Resampling.NEAREST)
            masks_resized.append(np.array(mask_resized))
        
        return img_resized, masks_resized
    
    def _pre_process_mask(self, mask):
        """Preprocess mask tensor."""
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        
        # Resize to CLIP input size (224x224)
        if mask.shape[-2:] != (224, 224):
            mask = F.interpolate(
                mask.unsqueeze(1).float(),
                size=(224, 224),
                mode='nearest'
            ).squeeze(1).long()
        
        return mask
    
    def crop_image_and_masks(self, I, masks):
        """Crop image to bounding boxes of masks."""
        crops = []
        
        for mask in masks:
            if isinstance(mask, np.ndarray):
                # Find bounding box
                rows = np.any(mask > 0, axis=1)
                cols = np.any(mask > 0, axis=0)
                
                if not rows.any() or not cols.any():
                    continue
                
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                # Add some padding
                pad = 5
                rmin = max(0, rmin - pad)
                rmax = min(mask.shape[0], rmax + pad)
                cmin = max(0, cmin - pad)
                cmax = min(mask.shape[1], cmax + pad)
                
                # Crop image and mask
                img_crop = I.crop((cmin, rmin, cmax, rmax))
                mask_crop = mask[rmin:rmax, cmin:cmax]
                
                crops.append((img_crop, mask_crop))
        
        return crops
    
    def process_tuple_list(self, crops):
        """Process list of (image, mask) tuples."""
        images = [c[0] for c in crops]
        masks = [c[1] for c in crops]
        return images, masks
