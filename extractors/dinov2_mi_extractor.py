"""
DINOv2 with Mask Inversion feature extractor.

This extractor combines DINOv2 with the Mask Inversion technique
for computing localized embeddings that focus on specific regions
of interest.
"""

import os
import sys

# Add the package directory to path for imports
_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PACKAGE_DIR not in sys.path:
    sys.path.insert(0, _PACKAGE_DIR)

import copy
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from extractors.dinov2_extractor import DinoV2Extractor
from utils.mask_inversion import MaskInversion


def get_dataset_class(args):
    """Get dataset class - imported here to avoid circular imports."""
    from datasets.base_dataset import get_dataset_class as _get_dataset_class
    return _get_dataset_class(args)


class DinoV2MIExtractor(DinoV2Extractor):
    """
    DINOv2 with Mask Inversion for localized feature extraction.
    
    This extractor:
    1. Uses DINOv2 as the backbone feature extractor
    2. Applies Mask Inversion to compute embeddings localized to mask regions
    3. Supports smart cropping for multi-object images
    """
    
    def __init__(self, args=None, device=None):
        super().__init__(args=args, device=device)
    
    def init_model(self):
        """Initialize DINOv2 model wrapped with MaskInversion."""
        # First initialize the base DINOv2 model
        super().init_model()
        
        # Wrap with MaskInversion
        lr = 0.1
        alpha = getattr(self.args, 'mi_alpha', 0.03)
        wd = 0.0
        iterations = getattr(self.args, 'mi_iterations', 100)
        layer_index = getattr(self.args, 'mi_layer_index', -1)
        
        self.base_model = self.model
        self.model = MaskInversion(
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
            (224, 224), 
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
        
        # Prepare for mask inversion
        full_img = None
        masks_all = None
        
        if getattr(self.args, 'mi_sum', False):
            full_img = self._pre_process_input(img_pil)
            if isinstance(full_img, dict):
                key = list(full_img.keys())[0]
                full_img = full_img[key]
            
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
                # Resize cropped images and masks to standard size
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
                alpha_sum_full=getattr(self.args, 'alpha_sum_full', 0),
                alpha_sum=getattr(self.args, 'alpha_sum', 0),
                verbose=getattr(self.args, 'debug', False),
                return_expl_map=True,
                full_mask=masks_all,
            )
        else:
            localized_embeddings, _, _ = self.model.compute_maskinversion(
                image=image_batch,
                masks_target=masks_tensor,
                verbose=getattr(self.args, 'debug', False),
                return_expl_map=True,
            )
        
        # Normalize embeddings
        localized_embeddings = F.normalize(localized_embeddings, dim=-1)
        
        return {
            'keypoints': localized_embeddings,
            'img_path': img_path
        }
    
    def _pre_process_mask(self, masks):
        """
        Preprocess masks for the model.
        
        Args:
            masks: Tensor of masks
            
        Returns:
            Preprocessed mask tensor
        """
        if masks.ndim == 2:
            masks = masks[None, None]
        elif masks.ndim == 3:
            masks = masks[:, None]
        
        masks = F.interpolate(
            masks.float(),
            size=(224, 224),
            mode='nearest'
        )
        
        return masks[:, 0]  # Remove channel dim


def get_extractor(args):
    """
    Factory function to get the appropriate extractor.
    
    Args:
        args: Configuration arguments with 'extractor' field
        
    Returns:
        Extractor instance, dataset class, dataset kwargs, output file path
    """
    extractor_name = getattr(args, 'extractor', 'dinov2_features')
    
    if 'dinov2_mi_features' in extractor_name:
        extractor = DinoV2MIExtractor(args=args)
    elif 'dinov2_features' in extractor_name:
        extractor = DinoV2Extractor(args=args)
    else:
        raise ValueError(f"Unknown extractor: {extractor_name}")
    
    # Get dataset info
    from datasets import get_dataset_class
    ds_class, ds_kwargs = get_dataset_class(args)
    
    # Build output file path
    add_string = getattr(args, 'add_string', '')
    if getattr(args, 'global_features', False):
        add_string += '_global'
    if getattr(args, 'smart_crop', False):
        add_string += '_smart_crop'
    if getattr(args, 'B_model', False):
        add_string += '_B_model'
    if getattr(args, 'mi_sum', False):
        add_string += '_mi_sum'
    if getattr(args, 'tag', ''):
        add_string += f"_{args.tag}"
    
    dataset = getattr(args, 'dataset', 'custom')
    out_file = f"outputs/{dataset}_retrieval/{dataset}_{extractor_name}{add_string}.pt"
    
    return extractor, ds_class, ds_kwargs, out_file
