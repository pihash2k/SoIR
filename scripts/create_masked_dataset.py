#!/usr/bin/env python3
"""
Create Masked Dataset Script

This script processes images using OWLv2 for object detection and 
SAM (Segment Anything Model) for segmentation to create masked datasets
suitable for object retrieval tasks.

Usage:
    python scripts/create_masked_dataset.py \
        --input_dir /path/to/images \
        --output_dir /path/to/output \
        --sam_checkpoint /path/to/sam_vit_h.pth \
        --owlv2_model google/owlv2-base-patch16-ensemble

Author: SoIR Team
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm
from torchvision import ops, transforms
from torchvision.transforms.functional import InterpolationMode
from typing import List, Dict, Optional, Tuple, Any


def install_dependencies():
    """Install required dependencies if not present."""
    import subprocess
    result = subprocess.run('pip list', shell=True, capture_output=True, text=True)
    installed_packages = [x.split(' ')[0] for x in result.stdout.split('\n')]
    
    if 'segment-anything' not in installed_packages:
        print("Installing segment-anything...")
        os.system('pip install git+https://github.com/facebookresearch/segment-anything.git')


# Try to import SAM, install if needed
try:
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    install_dependencies()
    from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator

try:
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
except ImportError:
    print("Installing transformers...")
    os.system('pip install transformers')
    from transformers import Owlv2Processor, Owlv2ForObjectDetection


def center_to_corners(bboxes_center: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes from center format to corner format.
    
    Args:
        bboxes_center: Tensor of shape (N, 4) with [cx, cy, w, h]
        
    Returns:
        Tensor of shape (N, 4) with [x1, y1, x2, y2]
    """
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        [(center_x - 0.5 * width),
         (center_y - 0.5 * height),
         (center_x + 0.5 * width),
         (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners


def post_process_owlv2_box_preds(boxes: torch.Tensor, image: Image.Image) -> torch.Tensor:
    """
    Post-process OWLv2 box predictions to absolute coordinates.
    
    Args:
        boxes: Predicted boxes in relative coordinates
        image: PIL Image for getting target size
        
    Returns:
        Boxes in absolute pixel coordinates
    """
    target_sizes = torch.tensor([max(image.size), max(image.size)])[None]

    # Convert to [x0, y0, x1, y1] format
    boxes = center_to_corners(boxes)

    # Convert from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(boxes.device)
    boxes = boxes * scale_fct[:, None, :]

    return boxes


def check_bb_is_inside(bb1: torch.Tensor, bb2: torch.Tensor, th_pr: float = 0.02) -> bool:
    """
    Check if bb1 is inside bb2.
    
    Args:
        bb1: First bounding box [x1, y1, x2, y2]
        bb2: Second bounding box [x1, y1, x2, y2]
        th_pr: Threshold proportion for margin
        
    Returns:
        True if bb1 is inside bb2
    """
    bbs = torch.cat((bb1[None], bb2[None]))
    large_bb = torch.argmax(ops.box_area(bbs))
    if large_bb != 1:
        return False
    small_bb = int(not large_bb.bool())
    small_large = bbs[small_bb] - bbs[large_bb]
    width = bbs[large_bb][2] - bbs[large_bb][0]
    height = bbs[large_bb][3] - bbs[large_bb][1]
    th_w = width * th_pr
    th_h = height * th_pr
    ths = torch.tensor([-th_w, -th_h, th_w, th_h]).to(small_large.device)
    ths_subtract = small_large - ths
    inside = torch.all(torch.logical_and(ths_subtract[:2] > 0, ths_subtract[2:] < 0))
    return inside


def post_process_bboxes(bboxes: torch.Tensor, scores: torch.Tensor, 
                        check_inside: bool = False) -> torch.Tensor:
    """
    Post-process bounding boxes with NMS and optional inside-check filtering.
    
    Args:
        bboxes: Detected bounding boxes
        scores: Confidence scores for each box
        check_inside: Whether to filter boxes that are inside other boxes
        
    Returns:
        Filtered bounding boxes
    """
    bboxes = bboxes.int().float()
    bb_inds = ops.nms(bboxes, scores, iou_threshold=0.4)
    bboxes = bboxes[bb_inds]
    
    if check_inside and bboxes.shape[0] > 1:
        inds_to_keep = []
        for bb_ind, bb in enumerate(bboxes):
            flag = True
            _boxes = torch.tensor([bboxes[_idx].tolist()
                                   for _idx in range(len(bboxes))
                                   if _idx != bb_ind]).to(bboxes.device)
            for _bb in _boxes:
                if check_bb_is_inside(bb, _bb):
                    flag = False
                    break
            if flag:
                inds_to_keep.append(bb_ind)
        inds_to_keep = torch.tensor(inds_to_keep)
        bboxes = bboxes[inds_to_keep]
    
    return bboxes


def draw_bounding_boxes(image: Image.Image, bboxes: torch.Tensor) -> Image.Image:
    """
    Draw bounding boxes on an image.

    Args:
        image: PIL.Image object
        bboxes: torch.Tensor of shape (N, 4) representing bounding boxes in xyxy format

    Returns:
        Image with bounding boxes drawn
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for bbox in bboxes:
        x0, y0, x1, y1 = bbox.int().tolist()
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    return image


class ObjectDetector:
    """OWLv2-based object detector."""
    
    def __init__(self, model_name: str = "google/owlv2-base-patch16-ensemble",
                 device: str = "cuda", objectness_threshold: float = 0.2):
        """
        Initialize the object detector.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Device to run inference on
            objectness_threshold: Minimum objectness score to keep detections
        """
        self.device = device
        self.objectness_threshold = objectness_threshold
        
        # Check if local path
        if os.path.exists(model_name):
            self.processor = Owlv2Processor.from_pretrained(
                model_name + '/processor', local_files_only=True
            )
            self.model = Owlv2ForObjectDetection.from_pretrained(
                model_name + '/model', local_files_only=True
            )
        else:
            self.processor = Owlv2Processor.from_pretrained(model_name)
            self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        
        self.model.to(device)
        self.model.eval()
    
    def detect(self, image: Image.Image, text_prompt: Optional[List[str]] = None
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect objects in an image.
        
        Args:
            image: PIL Image to process
            text_prompt: Optional text prompts for guided detection
            
        Returns:
            Tuple of (bboxes, scores)
        """
        if text_prompt is not None:
            return self._detect_with_text(image, text_prompt)
        else:
            return self._detect_objectness(image)
    
    def _detect_with_text(self, image: Image.Image, text_prompt: List[str]
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect objects using text guidance."""
        inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)

        W, H = image.size
        target_sizes = torch.Tensor([(max(W, H), max(W, H))])
        results = self.processor.post_process_object_detection(
            outputs=outputs, threshold=0.1, target_sizes=target_sizes
        )

        boxes = results[0]["boxes"]
        scores = results[0]["scores"]

        return boxes.cpu(), scores.cpu()
    
    def _detect_objectness(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Detect objects using objectness score only."""
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values']
        pixel_values = pixel_values.to(self.device)
        
        with torch.no_grad():
            feature_map, vision_outputs = self.model.image_embedder(
                pixel_values=pixel_values,
                output_attentions=False,
                output_hidden_states=False,
            )
            cur_shape = feature_map.shape
            feature_map_reshaped = feature_map.reshape(
                (cur_shape[0], cur_shape[1] * cur_shape[2], cur_shape[3])
            )
            box_preds = self.model.box_predictor(feature_map_reshaped, feature_map)
            box_preds = post_process_owlv2_box_preds(box_preds, image)
            scores = self.model.objectness_predictor(image_features=feature_map_reshaped)[0]
            scores = torch.sigmoid(scores)
            
            box_preds = box_preds[0, scores > self.objectness_threshold]
            scores = scores[scores > self.objectness_threshold]
            box_preds = post_process_bboxes(box_preds, scores, check_inside=False)

        return box_preds.cpu(), scores.cpu()


class Segmenter:
    """SAM-based image segmenter."""
    
    def __init__(self, checkpoint_path: str, model_type: str = "vit_h",
                 device: str = "cuda"):
        """
        Initialize the segmenter.
        
        Args:
            checkpoint_path: Path to SAM checkpoint
            model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to run inference on
        """
        self.device = device
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
    
    def segment(self, image: Image.Image, bboxes: Optional[torch.Tensor] = None
                ) -> List[np.ndarray]:
        """
        Segment objects in an image.
        
        Args:
            image: PIL Image to segment
            bboxes: Optional bounding boxes to guide segmentation
            
        Returns:
            List of binary masks
        """
        image_np = np.array(image)
        masks = []
        
        if bboxes is None or len(bboxes) == 0:
            # Automatic segmentation
            mask_results = self.mask_generator.generate(image_np)
            masks = [m['segmentation'] for m in mask_results]
        else:
            # Box-guided segmentation
            self.predictor.set_image(image_np)
            for bbox in bboxes:
                x0, y0, x1, y1 = bbox.int().tolist()
                input_box = np.array([[x0, y0, x1, y1]])
                mask_results, _, _ = self.predictor.predict(
                    box=input_box, multimask_output=False
                )
                masks.append(mask_results[0])
        
        return masks


def mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert binary mask to RLE encoding for efficient storage.
    
    Args:
        mask: Binary mask array
        
    Returns:
        Dictionary with 'counts' and 'size' keys
    """
    flat_mask = mask.flatten()
    changes = np.diff(flat_mask)
    change_indices = np.where(changes != 0)[0] + 1
    runs = np.diff(np.concatenate([[0], change_indices, [len(flat_mask)]]))
    
    # If mask starts with 1, prepend a 0 run
    if flat_mask[0] == 1:
        runs = np.concatenate([[0], runs])
    
    return {
        'counts': runs.tolist(),
        'size': list(mask.shape)
    }


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Convert RLE encoding back to binary mask.
    
    Args:
        rle: Dictionary with 'counts' and 'size' keys
        
    Returns:
        Binary mask array
    """
    counts = rle['counts']
    size = rle['size']
    
    mask = np.zeros(np.prod(size), dtype=np.uint8)
    current_pos = 0
    current_val = 0
    
    for count in counts:
        mask[current_pos:current_pos + count] = current_val
        current_pos += count
        current_val = 1 - current_val
    
    return mask.reshape(size)


def process_image(image_path: str, detector: ObjectDetector, 
                  segmenter: Segmenter, text_prompt: Optional[List[str]] = None
                  ) -> Dict[str, Any]:
    """
    Process a single image to extract masks and bboxes.
    
    Args:
        image_path: Path to the image
        detector: ObjectDetector instance
        segmenter: Segmenter instance
        text_prompt: Optional text prompts for detection
        
    Returns:
        Dictionary with masks, bboxes, and metadata
    """
    image = Image.open(image_path).convert("RGB")
    
    # Detect objects
    bboxes, scores = detector.detect(image, text_prompt)
    
    # Segment objects
    masks = segmenter.segment(image, bboxes)
    
    # Convert masks to RLE for storage
    masks_rle = [mask_to_rle(mask) for mask in masks]
    
    return {
        'image_path': str(image_path),
        'image_size': image.size,
        'bboxes': bboxes.tolist() if len(bboxes) > 0 else [],
        'scores': scores.tolist() if len(scores) > 0 else [],
        'masks_rle': masks_rle,
        'num_objects': len(masks)
    }


def process_dataset(input_dir: str, output_dir: str, 
                    sam_checkpoint: str, owlv2_model: str,
                    sam_model_type: str = "vit_h",
                    device: str = "cuda",
                    text_prompt: Optional[List[str]] = None,
                    save_visualizations: bool = True,
                    save_interval: int = 10) -> Dict[str, Any]:
    """
    Process a directory of images to create a masked dataset.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory for output files
        sam_checkpoint: Path to SAM checkpoint
        owlv2_model: OWLv2 model name or path
        sam_model_type: SAM model type
        device: Device for inference
        text_prompt: Optional text prompts for detection
        save_visualizations: Whether to save visualization images
        save_interval: Save intermediate results every N images
        
    Returns:
        Dictionary mapping image paths to results
    """
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    vis_path = output_path / "visualizations"
    if save_visualizations:
        vis_path.mkdir(exist_ok=True)
    
    # Initialize models
    print("Loading OWLv2 detector...")
    detector = ObjectDetector(owlv2_model, device)
    
    print("Loading SAM segmenter...")
    segmenter = Segmenter(sam_checkpoint, sam_model_type, device)
    
    # Find all images
    input_path = Path(input_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in input_path.rglob('*') 
                   if f.suffix.lower() in image_extensions]
    
    print(f"Found {len(image_files)} images to process")
    
    results = {}
    temp_path = output_path / "results.tmp.pt"
    final_path = output_path / "segmentation_results.pt"
    
    for idx, image_path in enumerate(tqdm(image_files, desc="Processing images")):
        try:
            result = process_image(str(image_path), detector, segmenter, text_prompt)
            results[str(image_path)] = result
            
            # Save visualization
            if save_visualizations and result['num_objects'] > 0:
                image = Image.open(image_path).convert("RGB")
                bboxes = torch.tensor(result['bboxes'])
                vis_image = draw_bounding_boxes(image, bboxes)
                vis_save_path = vis_path / f"{image_path.stem}_detections.jpg"
                vis_image.save(vis_save_path)
            
            # Periodic save
            if (idx + 1) % save_interval == 0:
                torch.save(results, temp_path)
                print(f"Saved intermediate results ({idx + 1}/{len(image_files)})")
                
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            results[str(image_path)] = {'error': str(e)}
    
    # Final save
    torch.save(results, final_path)
    if temp_path.exists():
        temp_path.unlink()
    
    print(f"Results saved to {final_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Create masked dataset using OWLv2 and SAM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Directory for output files"
    )
    
    parser.add_argument(
        "--sam_checkpoint",
        type=str,
        required=True,
        help="Path to SAM checkpoint (e.g., sam_vit_h_4b8939.pth)"
    )
    
    parser.add_argument(
        "--owlv2_model",
        type=str,
        default="google/owlv2-base-patch16-ensemble",
        help="OWLv2 model name or local path"
    )
    
    parser.add_argument(
        "--sam_model_type",
        type=str,
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference"
    )
    
    parser.add_argument(
        "--text_prompt",
        type=str,
        nargs="+",
        default=None,
        help="Optional text prompts for guided detection (e.g., 'a photo of a dog')"
    )
    
    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="Disable saving visualization images"
    )
    
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save intermediate results every N images"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sam_checkpoint=args.sam_checkpoint,
        owlv2_model=args.owlv2_model,
        sam_model_type=args.sam_model_type,
        device=args.device,
        text_prompt=args.text_prompt,
        save_visualizations=not args.no_visualizations,
        save_interval=args.save_interval
    )


if __name__ == "__main__":
    main()
