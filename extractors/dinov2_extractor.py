"""
DINOv2 feature extractor.
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
from collections import OrderedDict

from extractors.base_extractor import BaseExtractor

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

try:
    from peft import get_peft_model, LoraConfig
except ImportError:
    LoraConfig = None
    get_peft_model = None


def align_state_dicts(state_dict_with_backbone, state_dict_no_backbone, backbone_key="backbone"):
    """Align state dict keys between models with/without backbone prefix."""
    aligned_state_dict = OrderedDict()
    
    for key_with_backbone, value in state_dict_with_backbone.items():
        aligned_key = key_with_backbone.replace(f".{backbone_key}", "")
        if aligned_key in state_dict_no_backbone:
            aligned_state_dict[aligned_key] = value
    
    return aligned_state_dict


class DinoV2Extractor(BaseExtractor):
    """
    DINOv2 feature extractor for instance retrieval.
    
    Supports:
    - Base and Large model variants
    - LoRA fine-tuning
    - Global and local feature extraction
    - Caption-based mask extraction
    """
    
    def __init__(self, args=None, device=None):
        self.global_cls = True
        self.captions = None
        self.captions_file = None
        super().__init__(args=args, device=device)
    
    def init_model(self):
        """Initialize DINOv2 model and processor."""
        self.captions_file = getattr(self.args, 'captions_file', None)
        
        # Choose model variant
        if getattr(self.args, 'B_model', False):
            model_name = "facebook/dinov2-base"
            self.vec_dim = 768
        else:
            model_name = "facebook/dinov2-large"
            self.vec_dim = 1024
        
        print(f"Loading DINOv2 model: {model_name}")
        
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Disable center crop in processor
        self.processor.do_center_crop = False
        self.processor.size = {'height': 224, 'width': 224}
        
        self.model.to(self.device)
        
        # Load captions if provided
        if self.captions_file is not None:
            print(f"Loading captions file: {self.captions_file}")
            self.captions = torch.load(self.captions_file)
        
        # Load fine-tuned weights if provided
        weights_path = getattr(self.args, 'weights', None)
        if weights_path is not None:
            loaded_state_dict = torch.load(weights_path, map_location=self.device)
            if "model_state_dict" in loaded_state_dict:
                loaded_state_dict = loaded_state_dict["model_state_dict"]
            
            if getattr(self.args, 'lora_adapt', False):
                if LoraConfig is None or get_peft_model is None:
                    raise ImportError("Please install peft: pip install peft")
                
                r = getattr(self.args, 'lora_rank', 256)
                lora_alpha = getattr(self.args, 'lora_alpha', 32)
                
                lora_config = LoraConfig(
                    r=r,
                    lora_alpha=lora_alpha,
                    target_modules=["key", "query", "value"],
                    lora_dropout=0.1,
                    modules_to_save=None,
                )
                
                self.model = get_peft_model(self.model, lora_config)
                new_state_dict = align_state_dicts(
                    loaded_state_dict, self.model.state_dict()
                )
                self.model.load_state_dict(new_state_dict)
                print(f"Loaded LoRA weights from {weights_path}")
            else:
                self.model.load_state_dict(loaded_state_dict)
                print(f"Loaded weights from {weights_path}")
        
        self.model.eval()
    
    def _pre_process_input(self, image, is_mask=False):
        """
        Preprocess image for the model.
        
        Args:
            image: PIL Image or tensor
            is_mask: Whether input is a mask
            
        Returns:
            Preprocessed tensor dict
        """
        if is_mask:
            if isinstance(image, torch.Tensor):
                if image.ndim == 3:
                    image = image[None]
                inputs = F.interpolate(
                    image.float(),
                    size=(224, 224),
                    mode="nearest"
                )[0]
            else:
                # Assume numpy array
                image = torch.from_numpy(np.array(image))
                if image.ndim == 2:
                    image = image[None, None]
                elif image.ndim == 3:
                    image = image[None]
                inputs = F.interpolate(
                    image.float(),
                    size=(224, 224),
                    mode="nearest"
                )[0]
            return inputs
        
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: inputs[k].to(self.device) for k in inputs}
        return inputs
    
    def _call_model(self, inputs):
        """Call the model and get outputs."""
        with torch.no_grad():
            if isinstance(inputs, torch.Tensor):
                outputs = self.model(inputs)
            else:
                outputs = self.model(**inputs)
            
            if hasattr(outputs, 'pooler_output'):
                return {'pooler_output': outputs.pooler_output}
            elif hasattr(outputs, 'last_hidden_state'):
                # Use CLS token
                return {'pooler_output': outputs.last_hidden_state[:, 0]}
            else:
                return {'pooler_output': outputs}
    
    def extract(self, image, is_query=False, masks=None):
        """
        Extract features from an image.
        
        Args:
            image: PIL Image or dict with 'image' key
            is_query: Whether this is a query image
            masks: Optional masks for localized extraction
            
        Returns:
            Dict with 'keypoints' tensor
        """
        self.is_current_query = is_query
        
        # Handle dict input (for caption-based extraction)
        if isinstance(image, dict):
            img_path = image.get('img_path', None)
            img_pil = image['image']
        else:
            img_path = None
            img_pil = image
        
        # Preprocess image
        inputs = self._pre_process_input(img_pil)
        
        # Get features
        outputs = self._call_model(inputs)
        features = outputs['pooler_output']
        
        # Normalize features
        if not getattr(self.args, 'wo_norm_features', False):
            features = F.normalize(features, p=2, dim=-1)
        
        return {
            'keypoints': features,
            'img_path': img_path
        }
    
    def extract_mask_from_path(self, img_path, img_pil):
        """
        Extract masks from captions file for a given image.
        
        Args:
            img_path: Path to the image
            img_pil: PIL Image
            
        Returns:
            List of masks as numpy arrays
        """
        if self.captions is None or img_path not in self.captions:
            # Return full image mask
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        ann = self.captions[img_path]
        masks = []
        
        def decode_rle_mask(rle_dict):
            """Decode RLE mask from dictionary format."""
            h, w = rle_dict['size']
            counts = rle_dict['counts']
            
            if isinstance(counts, (bytes, str)):
                # Standard COCO RLE format
                from pycocotools import mask as mask_utils
                return mask_utils.decode(rle_dict)
            else:
                # Custom list-based RLE format
                mask = np.zeros((h * w,), dtype=np.uint8)
                idx = 0
                val = 0
                for c in counts:
                    end_idx = min(idx + c, h * w)
                    mask[idx:end_idx] = val
                    idx = end_idx
                    val = 1 - val
                mask = mask.reshape((h, w))
                
                if rle_dict.get('transpose', False):
                    mask = mask.T
                
                return mask
        
        # Handle different mask formats
        if 'masks' in ann:
            raw_masks = ann['masks']
            for m in raw_masks:
                if isinstance(m, dict) and 'counts' in m:
                    mask = decode_rle_mask(m)
                    masks.append(mask * 255)
                elif isinstance(m, np.ndarray):
                    masks.append(m)
                else:
                    masks.append(m)
        elif 'mask' in ann:
            # Single mask key (could be list of masks)
            mask_data = ann['mask']
            if not isinstance(mask_data, list):
                mask_data = [mask_data]
            for m in mask_data:
                if isinstance(m, dict) and 'counts' in m:
                    mask = decode_rle_mask(m)
                    masks.append(mask * 255)
                elif isinstance(m, np.ndarray):
                    masks.append(m)
                else:
                    masks.append(m)
        elif 'rle' in ann:
            # Decode RLE masks
            for rle in ann['rle']:
                mask = decode_rle_mask(rle)
                masks.append(mask * 255)
        elif 'bbox' in ann:
            # Create mask from bbox(es)
            bboxes = ann['bbox']
            # Handle single bbox vs list of bboxes
            if isinstance(bboxes, torch.Tensor):
                bboxes = bboxes.squeeze().tolist()
                if isinstance(bboxes[0], (int, float)):
                    bboxes = [bboxes]
            elif isinstance(bboxes, (list, tuple)):
                if len(bboxes) > 0 and isinstance(bboxes[0], (int, float)):
                    bboxes = [bboxes]  # Single bbox
            
            for bbox in bboxes:
                if isinstance(bbox, torch.Tensor):
                    bbox = bbox.tolist()
                mask = np.zeros((img_pil.height, img_pil.width), dtype=np.uint8)
                x1, y1, x2, y2 = map(int, bbox[:4])
                mask[y1:y2, x1:x2] = 255
                masks.append(mask)
        
        if len(masks) == 0:
            masks = [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        return masks
    
    def resize_image_and_masks(self, img_pil, masks, target_size=224):
        """
        Resize image and masks to target size.
        
        Args:
            img_pil: PIL Image
            masks: List of masks
            target_size: Target size
            
        Returns:
            Resized image and masks
        """
        # Resize image
        img_resized = img_pil.resize((target_size, target_size), Image.Resampling.LANCZOS)
        
        # Resize masks
        resized_masks = []
        for mask in masks:
            if isinstance(mask, np.ndarray):
                mask_pil = Image.fromarray(mask)
            else:
                mask_pil = mask
            mask_resized = mask_pil.resize((target_size, target_size), Image.Resampling.NEAREST)
            resized_masks.append(np.array(mask_resized))
        
        return img_resized, resized_masks
    
    def crop_image_and_masks(self, I, masks):
        """
        Smart crop: create individual crops for each mask region.
        
        Args:
            I: PIL Image
            masks: List of masks
            
        Returns:
            List of (cropped_image, cropped_mask) tuples
        """
        crops = []
        
        for mask in masks:
            if isinstance(mask, np.ndarray):
                mask_arr = mask
            else:
                mask_arr = np.array(mask)
            
            # Find bounding box of mask
            rows = np.any(mask_arr > 0, axis=1)
            cols = np.any(mask_arr > 0, axis=0)
            
            if not rows.any() or not cols.any():
                continue
            
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]
            
            # Add padding
            pad = int(max(y2 - y1, x2 - x1) * 0.1)
            y1 = max(0, y1 - pad)
            y2 = min(I.height, y2 + pad)
            x1 = max(0, x1 - pad)
            x2 = min(I.width, x2 + pad)
            
            # Crop
            img_crop = I.crop((x1, y1, x2, y2))
            mask_crop = mask_arr[y1:y2, x1:x2]
            
            crops.append((img_crop, mask_crop))
        
        return crops
    
    def process_tuple_list(self, crops):
        """
        Process list of (image, mask) tuples.
        
        Returns:
            List of images and list of masks
        """
        images = [c[0] for c in crops]
        masks = [c[1] for c in crops]
        return images, masks
