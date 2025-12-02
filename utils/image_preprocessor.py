"""
Image preprocessing utilities for the retrieval system.
"""

import copy
import numpy as np
from PIL import Image
import torch
import kornia


class ImagePreprocessor:
    """
    Preprocessor for preparing images before feature extraction.
    
    Handles:
    - Bounding box cropping
    - Resizing
    - Center cropping
    - Color space conversion
    """
    
    default_conf = {
        "resize": None,
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
        "grayscale": False,
    }

    def __init__(
        self,
        format='XYXY',
        center_crop=True,
        crop_no_resize=False,
        resize_query=True,
        grayscale=False,
        blur=None,
        blur_deblur_query=True,
        deblur=None,
        deblur_num=None,
        convert_colorspace=None,
        transform=None,
        args=None,
        partial=None
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            format: Bounding box format - 'XYXY', 'XYWH', or 'POLY'
            center_crop: Whether to center crop the image
            crop_no_resize: Crop without resizing
            resize_query: Whether to resize query images
            grayscale: Convert to grayscale
            blur: Blur filter to apply
            deblur: Sharpen filter to apply
            args: Additional arguments
            partial: Partial crop ratio (0-1)
        """
        self.format = format
        self.crop_no_resize = crop_no_resize
        self.center_crop = center_crop
        self.args = args
        self.blur = blur
        self.blur_deblur_query = blur_deblur_query
        self.deblur = deblur
        self.deblur_num = deblur_num
        self.transform = transform
        self.resize_query = resize_query
        self.partial = partial
        
        preprocess_conf = {
            **self.default_conf,
            "resize": 1024,
            "grayscale": grayscale,
            'convert_colorspace': convert_colorspace,
        }
        self.conf = preprocess_conf

    def __call__(self, img_pil, bbox=None, is_query=False, from_detector=False):
        """
        Preprocess an image.
        
        Args:
            img_pil: PIL Image
            bbox: Bounding box in the specified format
            is_query: Whether this is a query image
            from_detector: Whether bbox came from a detector
            
        Returns:
            Preprocessed PIL Image
        """
        if bbox is not None:
            img_pil = self._crop_by_bbox(img_pil, bbox, is_query, from_detector)
        
        # Apply blur/deblur if configured
        if self.blur is not None and self.blur_deblur_query == is_query:
            img_pil = img_pil.filter(self.blur)
        
        if self.deblur is not None and self.blur_deblur_query == is_query:
            for _ in range(self.deblur_num or 1):
                img_pil = img_pil.filter(self.deblur)
        
        return img_pil

    def _crop_by_bbox(self, img_pil, bbox, is_query=False, from_detector=False):
        """Crop image using bounding box."""
        # Handle nested bbox (e.g., tensor([[x1, y1, x2, y2]]))
        import torch
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.squeeze().tolist()
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 1:
            bbox = bbox[0]
            if isinstance(bbox, torch.Tensor):
                bbox = bbox.tolist()
        
        if self.format == 'XYXY':
            x1, y1, x2, y2 = bbox[:4]
        elif self.format == 'XYWH':
            x1, y1, w, h = bbox[:4]
            x2, y2 = x1 + w, y1 + h
        elif self.format == 'POLY':
            # For polygon, get bounding rectangle
            if isinstance(bbox, list) and isinstance(bbox[0], tuple):
                xs = [p[0] for p in bbox]
                ys = [p[1] for p in bbox]
            else:
                xs = bbox[::2]
                ys = bbox[1::2]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
        else:
            raise ValueError(f"Unknown bbox format: {self.format}")
        
        # Ensure valid coordinates
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_pil.width, int(x2)), min(img_pil.height, int(y2))
        
        if self.crop_no_resize:
            return img_pil.crop((x1, y1, x2, y2))
        
        # Center crop with padding
        if self.center_crop:
            return self._center_crop(img_pil, x1, y1, x2, y2)
        
        return img_pil.crop((x1, y1, x2, y2))

    def _center_crop(self, img_pil, x1, y1, x2, y2):
        """Create a center crop with the bbox content."""
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        # Make square based on larger dimension
        size = max(bbox_w, bbox_h)
        
        # Center the bbox in the square
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        new_x1 = int(cx - size / 2)
        new_y1 = int(cy - size / 2)
        new_x2 = int(cx + size / 2)
        new_y2 = int(cy + size / 2)
        
        # Handle padding if needed
        pad_left = max(0, -new_x1)
        pad_top = max(0, -new_y1)
        pad_right = max(0, new_x2 - img_pil.width)
        pad_bottom = max(0, new_y2 - img_pil.height)
        
        if pad_left or pad_top or pad_right or pad_bottom:
            # Create padded image
            padded = Image.new(
                img_pil.mode,
                (img_pil.width + pad_left + pad_right, 
                 img_pil.height + pad_top + pad_bottom),
                (0, 0, 0)
            )
            padded.paste(img_pil, (pad_left, pad_top))
            new_x1 += pad_left
            new_y1 += pad_top
            new_x2 += pad_left
            new_y2 += pad_top
            img_pil = padded
        
        return img_pil.crop((new_x1, new_y1, new_x2, new_y2))


def resize_by_scale(image, scale):
    """
    Resize image by a scale factor.
    
    Args:
        image: PIL Image
        scale: Scale factor
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
