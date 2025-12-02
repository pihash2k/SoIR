"""
SigLIP Feature Extractor.

Uses Google's SigLIP model for image feature extraction.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

from .base_extractor import BaseExtractor


class SigLIPExtractor(BaseExtractor):
    """
    SigLIP-based feature extractor.
    
    Supports:
    1. Global feature extraction
    2. Text prompt encoding
    3. Captions file for mask-based extraction
    """
    
    def __init__(self, args=None, device=None):
        super().__init__(args=args, device=device)
    
    def init_model(self):
        """Initialize SigLIP model."""
        # Load captions file first if specified
        self.captions_file = getattr(self.args, 'captions_file', None)
        self.captions = None
        if self.captions_file is not None:
            print(f"Loading captions file: {self.captions_file}")
            self.captions = torch.load(self.captions_file)
        
        # Choose model based on B_model flag
        if getattr(self.args, 'B_model', False):
            model_name = "google/siglip-base-patch16-384"
        else:
            model_name = "google/siglip-so400m-patch14-384"
        
        print(f"Loading SigLIP model: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = self.processor.tokenizer
        
        # Load custom weights if specified
        weights_path = getattr(self.args, 'weights', None)
        if weights_path is not None:
            self._load_weights(weights_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        # Get model parameters
        self.patch_size = self.model.vision_model.config.patch_size
        self.image_size = self.model.vision_model.config.image_size
    
    def _load_weights(self, weights_path):
        """Load custom weights."""
        print(f"Loading weights from {weights_path}")
        ckpt = torch.load(weights_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        
        # Check if LoRA weights
        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
        if lora_keys:
            self._add_lora_if_needed(state_dict)
        else:
            self.model.load_state_dict(state_dict, strict=False)
    
    def _add_lora_if_needed(self, state_dict):
        """Add LoRA adapters if needed."""
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_rank = getattr(self.args, 'lora_rank', 256)
            lora_alpha = getattr(self.args, 'lora_alpha', 32)
            
            # Find target modules from state dict
            target_modules = set()
            for key in state_dict.keys():
                if 'lora_A' in key or 'lora_B' in key:
                    # Extract module name
                    parts = key.split('.')
                    for i, part in enumerate(parts):
                        if part in ['query', 'key', 'value', 'dense', 'out_proj']:
                            target_modules.add(part)
            
            if not target_modules:
                target_modules = ["query", "key", "value"]
            
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                target_modules=list(target_modules),
                lora_dropout=0.0,
                bias="none",
            )
            
            self.model = get_peft_model(self.model, config)
            
            # Load the LoRA weights
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded LoRA weights (missing: {len(missing)}, unexpected: {len(unexpected)})")
            
            return True
        except ImportError:
            print("PEFT not available, skipping LoRA")
            return False
    
    def _pre_process_input(self, image):
        """Preprocess image for SigLIP."""
        if isinstance(image, torch.Tensor):
            return image
        
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)
    
    def _encode_text(self, texts):
        """Encode text using SigLIP text encoder."""
        inputs = self.processor(
            text=texts,
            images=None,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            input_ids = inputs['input_ids']
            if input_ids.shape[1] > 64:
                input_ids = input_ids[:, :64]
            text_features = self.model.text_model(input_ids)
        
        return text_features
    
    def _call_model(self, inputs):
        """Run forward pass through SigLIP vision model."""
        if isinstance(inputs, dict):
            pixel_values = inputs.get('pixel_values', inputs.get('x', None))
            if pixel_values is None:
                pixel_values = list(inputs.values())[0]
        else:
            pixel_values = inputs
        
        with torch.no_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
        
        # Return pooler output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif hasattr(outputs, 'last_hidden_state'):
            # Use mean of patch tokens (excluding CLS if present)
            return outputs.last_hidden_state.mean(dim=1)
        else:
            return outputs
    
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
        embeddings = self._call_model(inputs)
        
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        return {"keypoints": embeddings}
    
    def extract_mask_from_path(self, img_path, img_pil):
        """Extract masks from captions file."""
        if self.captions is None:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        if img_path not in self.captions:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        masks = self.captions[img_path].get('masks', [])
        if len(masks) == 0:
            return [np.ones((img_pil.height, img_pil.width), dtype=np.uint8) * 255]
        
        return masks
