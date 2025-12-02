"""
Example: Create a sample annotation file.

This script demonstrates the expected format for annotation files.
"""

import torch
from pathlib import Path


def create_sample_annotations(image_dir, output_path):
    """
    Create a sample annotation file from a directory of images.
    
    Expected image organization:
    image_dir/
        class1/
            query_image1.jpg
            gallery_image1.jpg
            ...
        class2/
            ...
    
    Or flat structure with naming convention:
    image_dir/
        class1_001.jpg  (query)
        class1_002.jpg  (gallery)
        class2_001.jpg  (query)
        ...
    """
    image_dir = Path(image_dir)
    annotations = {}
    
    # Example: Process images in subdirectories
    for class_dir in image_dir.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            class_id = hash(class_name) % 10000  # Simple class ID
            
            images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            
            for idx, img_path in enumerate(images):
                # First image is query, rest are gallery
                is_query = idx == 0
                
                annotations[str(img_path)] = {
                    "bbox": None,  # Or [x1, y1, x2, y2] if available
                    "ins": class_id,
                    "is_query": is_query,
                    "obj_name": class_name
                }
    
    # Save annotations
    torch.save(annotations, output_path)
    print(f"Saved {len(annotations)} annotations to {output_path}")
    
    return annotations


def create_manual_annotations():
    """
    Example of manually creating annotations.
    """
    annotations = {
        "/path/to/query1.jpg": {
            "bbox": [100, 100, 300, 300],  # x1, y1, x2, y2
            "ins": 0,  # Instance ID
            "is_query": True,
            "obj_name": "object_class_1"
        },
        "/path/to/gallery1.jpg": {
            "bbox": [50, 50, 250, 250],
            "ins": 0,  # Same instance as query1
            "is_query": False,
            "obj_name": "object_class_1"
        },
        "/path/to/gallery2.jpg": {
            "bbox": None,  # No bbox - use full image
            "ins": 0,
            "is_query": False,
            "obj_name": "object_class_1"
        },
        "/path/to/query2.jpg": {
            "bbox": [200, 150, 400, 350],
            "ins": 1,  # Different instance
            "is_query": True,
            "obj_name": "object_class_2"
        },
        # ... more images
    }
    
    return annotations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample annotations")
    parser.add_argument("--image_dir", type=str, help="Directory with images")
    parser.add_argument("--output", type=str, default="annotations.pt", help="Output path")
    
    args = parser.parse_args()
    
    if args.image_dir:
        create_sample_annotations(args.image_dir, args.output)
    else:
        # Show example
        print("Example annotation format:")
        annotations = create_manual_annotations()
        for path, ann in list(annotations.items())[:2]:
            print(f"\n{path}:")
            for k, v in ann.items():
                print(f"  {k}: {v}")
