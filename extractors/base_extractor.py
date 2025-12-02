"""
Base extractor class for feature extraction.
"""

import torch
from abc import ABC, abstractmethod
from PIL import Image


class BaseExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    All extractors should inherit from this class and implement
    the extract() method.
    """
    
    def __init__(self, args=None, device=None):
        """
        Initialize the extractor.
        
        Args:
            args: Configuration arguments
            device: Device to use for computation
        """
        self.args = args
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_prompt = None
        self.is_current_query = False
        self.init_model()
    
    @abstractmethod
    def init_model(self):
        """Initialize the model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def extract(self, image, is_query=False, masks=None):
        """
        Extract features from an image.
        
        Args:
            image: PIL Image or dict with 'image' and optionally 'img_path'
            is_query: Whether this is a query image
            masks: Optional masks for localized extraction
            
        Returns:
            Dict with 'keypoints' tensor and other metadata
        """
        pass
    
    def set_txt_prompt(self, prompt):
        """Set text prompt for text-guided extraction."""
        self.text_prompt = prompt
    
    def _extract_global_features(self, images, is_query=False):
        """
        Extract global features from a list of images.
        
        Args:
            images: List of PIL Images or image dicts
            is_query: Whether these are query images
            
        Returns:
            Tensor of global features [N, D]
        """
        if not isinstance(images, list):
            images = [images]
        
        features = []
        for img in images:
            self.is_current_query = is_query
            feat = self.extract(img, is_query=is_query)
            features.append(feat['keypoints'])
        
        return torch.cat(features, dim=0) if features else torch.tensor([])
    
    def merge_multiple_vecs(self, keypoints, key='keypoints', img_path=None):
        """
        Merge multiple vectors into a single representation.
        
        Default implementation: average pooling.
        
        Args:
            keypoints: Dict with 'keypoints' tensor
            key: Key to access in the dict
            img_path: Optional image path for context
            
        Returns:
            Dict with merged keypoints
        """
        kpts = keypoints[key]
        if kpts.ndim > 1 and kpts.shape[0] > 1:
            merged = kpts.mean(dim=0, keepdim=True)
        else:
            merged = kpts
        return {**keypoints, key: merged}
