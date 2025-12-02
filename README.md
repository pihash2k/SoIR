# SoIR - Small Object Image Retrieval

A standalone instance retrieval system with **Mask Inversion** for computing localized embeddings. Supports multiple vision foundation models: **DINOv2**, **SigLIP**, and **CLIP**.

## Overview

This repository provides a complete pipeline for salient object instance retrieval:

1. **Dataset Preparation**: Create masked datasets using OWLv2 detection + SAM segmentation
2. **Feature Extraction**: Extract localized embeddings using Mask Inversion
3. **Index Creation**: Build FAISS indices for efficient similarity search
4. **Retrieval & Evaluation**: Search and compute retrieval metrics (mAP, Recall@K)

### Supported Extractors

| Extractor | Model | Embedding Dim | Expected Performance |
|-----------|-------|---------------|---------------------|
| `dinov2_mi_features` | DINOv2-Base | 768 | mAP ~0.35, R@1 ~74% |
| `siglip_mi_features` | SigLIP-Base-384 | 768 | mAP ~0.86, R@1 ~87% |
| `clip_mi_features` | CLIP ViT-B/16 | 512 | mAP ~0.78, R@1 ~83% |

## Installation

```bash
# Clone the repository
git clone https://github.com/pihash2k/SoIR.git
cd SoIR

# Install dependencies
pip install -r requirements.txt

# For mask dataset creation (optional)
pip install segment-anything
```

## Directory Structure

```
SoIR/
├── README.md
├── requirements.txt
├── configs/
│   └── defaults.yaml              # Default configuration
├── create_index.py                # Create FAISS index from gallery
├── search_index.py                # Search index and evaluate
├── extractors/
│   ├── __init__.py
│   ├── base_extractor.py          # Abstract base class
│   ├── dinov2_extractor.py        # DINOv2 feature extractor
│   ├── dinov2_mi_extractor.py     # DINOv2 + Mask Inversion
│   ├── siglip_extractor.py        # SigLIP feature extractor
│   ├── siglip_mi_extractor.py     # SigLIP + Mask Inversion
│   ├── clip_extractor.py          # CLIP feature extractor
│   └── clip_mi_extractor.py       # CLIP + Mask Inversion
├── datasets/
│   ├── __init__.py
│   └── base_dataset.py            # Dataset loader
├── utils/
│   ├── __init__.py
│   ├── image_preprocessor.py      # Image preprocessing
│   ├── mask_inversion.py          # Mask Inversion implementation
│   └── metrics.py                 # mAP, Recall metrics
├── scripts/
│   └── create_masked_dataset.py   # OWLv2 + SAM mask creation
└── example/
    └── run_example.sh             # Example usage script
```

## Quick Start

### 1. Create FAISS Index

```bash
# Using SigLIP (best performance)
python create_index.py \
    dataset=instre \
    extractor=siglip_mi_features \
    vec_dim=768 \
    mi_alpha=0.03 \
    global_features=true \
    anns_file=/path/to/annotations.pt

# Using CLIP
python create_index.py \
    dataset=instre \
    extractor=clip_mi_features \
    vec_dim=512 \
    mi_alpha=0.03 \
    global_features=true \
    anns_file=/path/to/annotations.pt

# Using DINOv2 with LoRA
python create_index.py \
    dataset=instre \
    extractor=dinov2_mi_features \
    B_model=true \
    vec_dim=768 \
    mi_alpha=0.03 \
    lora_adapt=true \
    lora_rank=256 \
    weights=/path/to/lora_weights.ckpt \
    anns_file=/path/to/annotations.pt
```

### 2. Search and Evaluate

```bash
python search_index.py \
    dataset=instre \
    experiment=my_experiment \
    k_search=100 \
    features_dir=/path/to/features
```

### 3. Create Masked Dataset (Optional)

```bash
python scripts/create_masked_dataset.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --owlv2_model google/owlv2-base-patch16-ensemble
```

## Configuration Reference

### Main Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset` | Dataset name (used for paths) | `pe` |
| `extractor` | Feature extractor type | `siglip_mi_features` |
| `vec_dim` | Embedding dimension | `768` |
| `mi_alpha` | Mask Inversion regularization weight | `0.03` |
| `global_features` | Extract global (CLS) features | `true` |
| `smart_crop` | Use smart cropping with masks | `false` |

### DINOv2-Specific Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `B_model` | Use DINOv2-Base (vs Large) | `true` |
| `lora_adapt` | Enable LoRA fine-tuning | `false` |
| `lora_rank` | LoRA rank | `256` |
| `weights` | Path to LoRA weights checkpoint | `null` |

### Dataset Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `anns_file` | Path to annotations .pt file | Required |
| `captions_file` | Path to captions/masks .pt file | `null` |
| `galleries_dir` | Directory with gallery images | Auto |
| `queries_dir` | Directory with query images | Auto |

## Dataset Format

### Annotation File (`anns.pt`)

```python
{
    "/path/to/image1.jpg": {
        "bbox": [x1, y1, x2, y2],      # Optional bounding box
        "ins": 0,                       # Instance/class label (int)
        "is_query": True,               # True for queries, False for gallery
        "obj_name": "eiffel_tower"      # Optional object name
    },
    "/path/to/image2.jpg": {
        "ins": 0,
        "is_query": False
    },
    ...
}
```

### Captions/Masks File (`captions.pt`) - Optional

Used for smart cropping with pre-computed segmentation masks:

```python
{
    "/path/to/image.jpg": {
        "masks_rle": [
            {"counts": [...], "size": [H, W]},  # RLE-encoded masks
            ...
        ],
        "bboxes": [[x1, y1, x2, y2], ...],
        "scores": [0.95, 0.87, ...]
    },
    ...
}
```

## Mask Inversion

Mask Inversion is a technique for computing attention-weighted feature embeddings that focus on salient objects. It works by:

1. **Forward Pass**: Extract intermediate features from vision transformer
2. **Attention Computation**: Compute attention maps between CLS token and patches
3. **Gradient-based Saliency**: Use gradients to weight feature importance
4. **Weighted Pooling**: Pool features using the computed attention/saliency maps

The `mi_alpha` parameter controls regularization strength - higher values produce more uniform attention, lower values focus more strongly on salient regions.

## Example Results

On the INSTRE dataset:

| Method | mAP | Recall@1 | Recall@5 | Recall@10 |
|--------|-----|----------|----------|-----------|
| SigLIP-MI | **0.86** | **86.79%** | 90.57% | 92.45% |
| CLIP-MI | 0.78 | 83.02% | 88.68% | 90.57% |
| DINOv2-MI | 0.35 | 74.53% | 79.25% | 83.02% |

## API Usage

```python
from extractors import SigLIPMIExtractor, CLIPMIExtractor, DINOv2MIExtractor
from PIL import Image

# Initialize extractor
extractor = SigLIPMIExtractor(
    mi_alpha=0.03,
    global_features=True
)

# Load and preprocess image
image = Image.open("image.jpg")

# Extract features
features = extractor.extract(image)  # Shape: (768,)
```

## Citation

If you use this code, please cite:

```bibtex
@article{soir2024,
    title={SoIR: Salient Object Instance Retrieval with Mask Inversion},
    year={2024}
}
```

## License

MIT License

## Acknowledgments

- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-384) by Google
- [CLIP](https://github.com/openai/CLIP) by OpenAI
- [Segment Anything](https://github.com/facebookresearch/segment-anything) by Meta AI
- [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) by Google
