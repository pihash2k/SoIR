"""
CLIP feature extractor using OpenAI's CLIP model.
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

from extractors.base_extractor import BaseExtractor

try:
    import clip
except ImportError:
    raise ImportError("Please install clip: pip install git+https://github.com/openai/CLIP.git")


class CLIPExtractor(BaseExtractor):
    """
    CLIP feature extractor for instance retrieval.
    
    Uses OpenAI's CLIP ViT-B/16 model for feature extraction.
    """
    
    def __init__(self, args=None, device=None):
        self.global_cls = True
        self.captions = None
        self.captions_file = None
        super().__init__(args=args, device=device)
        
        # Load captions if specified
        if hasattr(self.args, 'captions_file') and self.args.captions_file:
            self.captions_file = self.args.captions_file
            print(f"Loading captions file: {self.captions_file}")
            self.captions = torch.load(self.captions_file)
    
    def init_model(self):
        """Initialize CLIP model."""
        print("Loading CLIP model: ViT-B/16")
        self.model, self.processor = clip.load("ViT-B/16", device=self.device)
        self.model.eval()
        self.model.float()  # Ensure float32 for compatibility
        
        # Remove center crop from transforms
        from torchvision.transforms import CenterCrop
        self.processor.transforms = [
            t for t in self.processor.transforms
            if not isinstance(t, CenterCrop)
        ]
        # Set to 224x224 (CLIP default)
        self.processor.transforms[0].size = (224, 224)
        
        # Tokenizer for text encoding
        self.tokenizer = clip.tokenize
    
    def _encode_text(self, texts):
        """Encode text prompts to embeddings."""
        text_tokens = self.tokenizer(texts, truncate=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
        return text_features.float()
    
    def _pre_process_input(self, image, is_mask=False):
        """Preprocess image for CLIP."""
        if is_mask:
            original_dtype = image.dtype
            if image.ndim == 3:
                image = image[None]
            inputs = torch.nn.functional.interpolate(
                image.float(),
                size=(224, 224),
                mode="nearest",
            ).to(original_dtype)[0]
        else:
            inputs = self.processor(image).unsqueeze(0).to(self.device)
        return inputs
    
    def _call_model(self, inputs):
        """Run forward pass through CLIP image encoder."""
        with torch.no_grad():
            if isinstance(inputs, dict):
                _k = list(inputs.keys())[0]
                inputs = inputs[_k]
            outputs = self.model.encode_image(inputs)
            outputs = outputs.float()
        return {"pooler_output": outputs}
    
    def extract(self, image, is_query=False, masks=None):
        """
        Extract features from an image.
        
        Args:
            image: PIL Image or dict with 'image' and 'img_path' keys
            is_query: Whether this is a query image
            masks: Optional pre-computed masks
            
        Returns:
            Dict with 'keypoints' tensor
        """
        self.is_current_query = is_query
        
        # Handle dict input
        if isinstance(image, dict):
            img_path = image.get('img_path', None)
            img_pil = image['image']
        else:
            img_path = None
            img_pil = image
        
        # Preprocess
        inputs = self._pre_process_input(img_pil)
        
        # Extract features
        outputs = self._call_model(inputs)
        embeddings = outputs.get('pooler_output', outputs)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return {"keypoints": embeddings}
    
    def extract_mask_from_path(self, img_path, img_pil):
        """Extract masks from captions file."""
        if self.captions is None:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        # Get the key from the path
        key = os.path.basename(img_path)
        
        # Try different key formats
        if key not in self.captions:
            key = img_path
        if key not in self.captions:
            # Try relative path
            for k in self.captions.keys():
                if key in k or k in img_path:
                    key = k
                    break
        
        if key not in self.captions:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        caption_data = self.captions[key]
        
        # Handle different caption data formats
        if isinstance(caption_data, dict):
            if 'masks' in caption_data:
                masks_data = caption_data['masks']
            elif 'segmentation' in caption_data:
                masks_data = caption_data['segmentation']
            else:
                return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        elif isinstance(caption_data, (list, np.ndarray)):
            masks_data = caption_data
        else:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        # Convert to list of numpy arrays
        masks = []
        for mask_item in masks_data:
            if isinstance(mask_item, dict):
                # RLE encoded mask
                if 'counts' in mask_item and 'size' in mask_item:
                    from utils.mask_utils import decode_rle_mask
                    mask = decode_rle_mask(mask_item)
                    masks.append(mask)
                elif 'mask' in mask_item:
                    masks.append(np.array(mask_item['mask']))
            elif isinstance(mask_item, np.ndarray):
                masks.append(mask_item)
            elif isinstance(mask_item, torch.Tensor):
                masks.append(mask_item.cpu().numpy())
        
        if len(masks) == 0:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        return masks
