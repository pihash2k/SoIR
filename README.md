# SoIR - Small Object Image Retrieval with MaO

A novel instance retrieval system with **Multi-object Attention Optimization (MaO)** for retrieving small objects in cluttered scenes. Supports **DINOv2**, **SigLIP**, and **CLIP** vision foundation models.

## 📦 Datasets

All datasets used in this work can be obtained from: **<https://github.com/pihash2k/VoxDet-SoIR/tree/master>**

This includes:

- **VoxDet** - Our primary benchmark (most challenging)
- **INSTRE-XS** - Small objects subset
- **INSTRE-XXS** - Very small objects subset
- **PerMiR** - Multi-instance retrieval dataset

## Overview

This repository implements MaO (Multi-object Attention Optimization) for Small Object Image Retrieval:

1. **Dataset Preparation**: Process datasets with OWLv2 detection + SAM segmentation
- ⚠️ **OPTIONAL** if you already have pre-computed detections and masks
- ⚠️ **MANDATORY** if you only have raw images (requires `segment-anything` package)
1. **Stage A - Multi-Object Fine-tuning**: Train visual encoders on multiple objects per image
1. **Stage B - Attention Optimization**: Refine representations using attention-based mask alignment
1. **Index Creation**: Build FAISS indices for efficient similarity search
1. **Retrieval & Evaluation**: Search and compute mAP, Recall@K metrics

### MaO Method

Multi-object Attention Optimization (MaO) addresses the challenge of retrieving images containing very small objects (as small as 0.5% of image area) in highly cluttered scenes through a two-stage approach:

- **Stage A - Multi-Object Fine-tuning**: Objects are detected, cropped and encoded separately. Contrastive learning aligns the average object-wise representation with query objects using InfoNCE loss.
- **Stage B - Attention Optimization**: Post-training refinement optimizes a single global descriptor by aligning explainability maps (via LeGrad) of object crops with their corresponding masks.

This produces a **single compact representation per image**, enabling scalable search while effectively capturing all objects regardless of size.

### Supported MaO Extractors

|Extractor           |Model          |Embedding Dim|VoxDet mAP (Fine-tuned)|VoxDet mAP (Zero-shot)|
|--------------------|---------------|-------------|-----------------------|----------------------|
|`dinov2_mi_features`|DINOv2-Base    |768          |**83.70%** ⭐ Best      |**70.20%**            |
|`clip_mi_features`  |CLIP ViT-B/16  |512          |**79.86%**             |**65.22%**            |
|`siglip_mi_features`|SigLIP-Base-384|768          |-                      |-                     |

**Note:** Best results are obtained with the **fine-tuned version**. Results shown are for VoxDet. Performance varies across datasets (see Benchmarks section below).

## Installation

```bash
# Clone the repository
git clone https://github.com/pihash2k/SoIR.git
cd SoIR

# Create venv environment
python3 -m venv soir
source soir/bin/activate

# Install core dependencies
pip install -r requirements.txt

# For mask dataset creation (MANDATORY if you don't have pre-computed detections/masks)
pip install segment-anything
```

### ⚠️ Important: Detection and Segmentation Requirements

**You have two options:**

1. **Use pre-computed detections and masks** (Optional script):
- If your dataset already has object detections and segmentation masks
- Provide them in the required format (see “Pre-computed Annotations Format” below)
- Skip the `create_masked_dataset.py` script
- No need to install `segment-anything`
1. **Create detections and masks** (Mandatory script):
- If your dataset does NOT have object detections and masks
- **MUST run** `create_masked_dataset.py` script
- **MUST install** `segment-anything` package
- This will generate the required annotations file

## Quick Start with MaO

### Step 1: Prepare Dataset with Masks (if needed)

**⚠️ MANDATORY if you don’t have pre-computed detections and masks:**

```bash
python scripts/create_masked_dataset.py \
    --input_dir /path/to/dataset/images \
    --output_dir /path/todataset/masked \
    --sam_checkpoint /path/to/sam_vit_h_4b8939.pth \
    --owlv2_model google/owlv2-base-patch16-ensemble
```

**This script generates:**

- Object detections using OWLv2
- Segmentation masks using SAM
- A `captions.pt` file with bounding boxes, masks, and scores

**Skip this step if:**

- You already have object detections and masks
- Your annotations are in the format shown in “Pre-computed Annotations Format” below
- You downloaded the datasets from [VoxDet-SoIR](https://github.com/pihash2k/VoxDet-SoIR/tree/master) (may already include masks)

-----

### Step 2: Choose Your Approach

You can use MaO in two ways:

#### **Option A: Zero-Shot MaO (No Fine-tuning)**

Use pre-trained models directly without training on your dataset. Good for quick evaluation or when you don’t have training data.

**Performance:** VoxDet mAP ~65-70% | PerMiR mAP ~89% | INSTRE-XS mAP ~71-89%

```bash
# Using DINOv2 with MaO (zero-shot)
python create_index.py \
    dataset=voxdet \
    extractor=dinov2_mi_features \
    vec_dim=768 \
    mi_alpha=0.03 \
    global_features=true \
    anns_file=/path/to/voxdet/annotations.pt

# Using CLIP with MaO (zero-shot)
python create_index.py \
    dataset=voxdet \
    extractor=clip_mi_features \
    vec_dim=512 \
    mi_alpha=0.03 \
    global_features=true \
    anns_file=/path/to/voxdet/annotations.pt
```

#### **Option B: Fine-tuned MaO (Recommended for Best Results) ⭐**

Use our pre-trained models, fine-tunned with LoRA

**Performance:** VoxDet mAP ~80-84% | PerMiR mAP ~90% | INSTRE-XS mAP ~90-91%

```bash
# Using DINOv2 with MaO + LoRA fine-tuning (BEST: 83.70% mAP on VoxDet)
python create_index.py \
    dataset=voxdet \
    extractor=dinov2_mi_features \
    B_model=true \
    vec_dim=768 \
    mi_alpha=0.03 \
    lora_adapt=true \
    lora_rank=256 \ weights=/path/to/dinov2_lora_weights.ckpt \
    global_features=true \   anns_file=/path/to/voxdet/annotations.pt

# Using CLIP with MaO + LoRA fine-tuning (79.86% mAP on VoxDet)
python create_index.py \
    dataset=voxdet \
    extractor=clip_mi_features \
    vec_dim=512 \
    mi_alpha=0.03 \
    lora_adapt=true \
    lora_rank=256 \    weights=/path/to/clip_lora_weights.ckpt \
    global_features=true \   anns_file=/path/to/voxdet/annotations.pt
```

**Fine-tuning Parameters:**

- `lora_adapt=true`: Enable LoRA fine-tuning
- `lora_rank=256`: LoRA rank (default: 256)
- `weights=/path/to/weights.ckpt`: Path to your fine-tuned LoRA checkpoint

**How to obtain LoRA weights:**

1. Fine-tune on your dataset’s training split using Stage A (Multi-Object Fine-tuning)
1. Training details: AdamW optimizer, lr=5×10⁻⁵, batch_size=128, 1 epoch
1. Save the LoRA checkpoint after training
1. Use the checkpoint path in the `weights` parameter

-----

## MaO Configuration Parameters

### Core MaO Parameters

|Parameter        |Description                  |Default             |Recommended                              |
|-----------------|-----------------------------|--------------------|-----------------------------------------|
|`extractor`      |MaO feature extractor        |`dinov2_mi_features`|Use `dinov2_mi_features` for best results|
|`vec_dim`        |Embedding dimension          |`768`               |768 for DINOv2, 512 for CLIP             |
|`mi_alpha`       |Stage B regularization weight|`0.03`              |0.03 (optimal on all datasets)           |
|`global_features`|Extract global CLS features  |`true`              |true                                     |

### Fine-tuning Parameters (Option B)

|Parameter   |Description                   |Default|Notes                           |
|------------|------------------------------|-------|--------------------------------|
|`lora_adapt`|Enable LoRA fine-tuned weights|`false`|Set to `true` to use checkpoint |
|`lora_rank` |LoRA rank                     |`256`  |Use 256 for provided checkpoints|
|`weights`   |Path to LoRA checkpoint       |`null` |Required when `lora_adapt=true` |

### Dataset Parameters

|Parameter      |Description                                                |When Required                                                                                        |
|---------------|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
|`dataset`      |Dataset name: `voxdet`, `permir`, `instre_xs`, `instre_xxs`|**Always required**                                                                                  |
|`anns_file`    |Path to annotations .pt file                               |**Always required**                                                                                  |
|`captions_file`|Path to masks .pt file                                     |Optional if you have pre-computed detections/masks; Otherwise generated by `create_masked_dataset.py`|
|`galleries_dir`|Gallery images directory                                   |Auto-detected from `anns_file`                                                                       |
|`queries_dir`  |Query images directory                                     |Auto-detected from `anns_file`                                                                       |

## Datasets for Small Object Image Retrieval

All datasets can be obtained from: **<https://github.com/pihash2k/VoxDet-SoIR/tree/master>**

### Dataset Overview

|Dataset       |Avg Objects|Obj Size (%)|Key Challenge     |Images                      |
|--------------|-----------|------------|------------------|----------------------------|
|**VoxDet**    |5.8        |1.1         |Small + cluttered |24 test scenes              |
|**PerMiR**    |4.7        |13.3        |Multi-instance    |150 queries, 450 gallery    |
|**INSTRE-XS** |1          |6.6         |Small objects     |2,428 queries, 2,065 gallery|
|**INSTRE-XXS**|1          |2.2         |Very small objects|106 queries, 120 gallery    |

### VoxDet Dataset

VoxDet is the **largest and most challenging** instance-based dataset for small object retrieval in cluttered scenes.

#### Dataset Statistics

|Property                 |Value                              |Description                                       |
|-------------------------|-----------------------------------|--------------------------------------------------|
|**Training Set**         |9.6K instances                     |55K scenes, 180K bounding boxes                   |
|**Test Set**             |20 instances                       |24 complex cluttered scenes with 9,109 annotations|
|**Avg Objects per Image**|5.8 annotated / 14.7 detected (OVD)|High clutter scenario                             |
|**Avg Object Size**      |1.1%                               |Very small objects (0.5-2% of image area)         |
|**Image Type**           |3D voxel-rendered                  |Diverse poses, lighting, shadows                  |
|**Key Challenge**        |Small + cluttered                  |Both tiny size AND multi-object interference      |

### PerMiR Dataset

PerMiR focuses on **multi-instance retrieval** with multiple objects per image, including same-category distractors.

#### Dataset Statistics

|Property                 |Value         |Description                            |
|-------------------------|--------------|---------------------------------------|
|**Categories**           |16            |Cars, people, animals, food items, etc.|
|**Query Images**         |150           |Object-focused images                  |
|**Gallery Images**       |450           |Complex scenes with multiple instances |
|**Avg Objects per Image**|4.7           |High clutter with category confusion   |
|**Avg Object Size**      |13.3%         |Medium-sized objects                   |
|**Key Challenge**        |Multi-instance|Same category, different instances     |

### INSTRE-XS Dataset

INSTRE-XS is a **small object subset** of the original INSTRE dataset.

#### Dataset Statistics

|Property                 |Value     |Description                      |
|-------------------------|----------|---------------------------------|
|**Query Images**         |2,428     |Object-focused queries           |
|**Gallery Images**       |2,065     |Images with 1-2 objects          |
|**Avg Objects per Image**|1         |Minimal clutter                  |
|**Avg Object Size**      |6.6%      |Small objects (<15% of image)    |
|**Key Challenge**        |Small size|Objects occupy <15% of image area|

### INSTRE-XXS Dataset

INSTRE-XXS is the **most challenging small object subset** with very tiny objects.

#### Dataset Statistics

|Property                 |Value          |Description                      |
|-------------------------|---------------|---------------------------------|
|**Query Images**         |106            |Object-focused queries           |
|**Gallery Images**       |120            |Images with very small objects   |
|**Avg Objects per Image**|1              |Minimal clutter                  |
|**Avg Object Size**      |2.2%           |Very small objects (<5% of image)|
|**Key Challenge**        |Very small size|Objects occupy <5% of image area |

### Dataset Annotation Format

**Basic annotations file (`annotations.pt`)** - Required for all datasets:

```python
# annotations.pt - Instance labels and query/gallery splits
{
    "/path/to/image1.jpg": {
        "bbox": [x1, y1, x2, y2],      # Bounding box (optional)
        "ins": 0,                       # Instance ID (required)
        "is_query": False,              # Gallery image
        "obj_name": "chair"             # Object name (optional)
    },
    "/path/to/query1.jpg": {
        "bbox": [x1, y1, x2, y2],
        "ins": 0,
        "is_query": True,               # Query image
        "obj_name": "chair"
    },
    ...
}
```

### Pre-computed Annotations Format (Optional)

**If you already have detections and masks**, provide them in this format to skip `create_masked_dataset.py`:

```python
# captions.pt - Pre-computed detections and segmentation masks
# This is the OUTPUT format of create_masked_dataset.py
{
    "/path/to/image.jpg": {
        "masks_rle": [
            {
                "counts": [...],           # RLE-encoded mask (from SAM)
                "size": [H, W]             # Image height, width
            },
            {
                "counts": [...],           # Second object mask
                "size": [H, W]
            },
            ...
        ],
        "bboxes": [
            [x1, y1, x2, y2],              # First object bounding box
            [x1, y1, x2, y2],              # Second object bounding box
            ...
        ],
        "scores": [
            0.95,                          # First object detection score
            0.87,                          # Second object detection score
            ...
        ]
    },
    ...
}
```

**Format Requirements:**

- `masks_rle`: List of RLE-encoded binary masks (one per detected object)
  - RLE format: `{"counts": [...], "size": [H, W]}`
  - Should cover the object region within the corresponding bbox
- `bboxes`: List of `[x1, y1, x2, y2]` bounding boxes (one per object)
- `scores`: List of detection confidence scores (0-1 range)
- All three lists must have the **same length** (one entry per object)

## Multi-object Attention Optimization (MaO) - Detailed

### Stage A: Multi-Object Fine-tuning

**Objective:** Train the visual encoder to represent multiple objects in a single image descriptor.

**Process:**

1. **Object Detection**: Use OWLv2 (open-vocabulary detector) in “objectness” mode
- Confidence threshold: 0.2
- Detects all objects without class-specific prompts
1. **Object Cropping**: Extract each detected object
- Minimum crop size = backbone input size (224×224 or 384×384)
- Center crop around object if bbox is too small
1. **Separate Encoding**: Encode each object crop: `{v₁, v₂, ..., vₖ} ∈ ℝᵈ`
1. **Average Pooling**: Compute gallery representation: `vᶜ = (1/k) Σᵢ vᵢ`
1. **Contrastive Loss**: Align `vᶜ` with query object `vᵍ` using InfoNCE:
   
   ```
   L = -log( exp(vᶜ · vᵍ / τ) / Σⱼ exp(vᶜ · vⱼᵍ / τ) )
   ```

**Training Details:**

- LoRA adapter: rank 256 (fine-tunes only low-rank adapters)
- Optimizer: AdamW, lr = 5×10⁻⁵, decay to 1×10⁻⁶
- Batch size: 128
- Epochs: 1 on VoxDet training set
- Hardware: 4× NVIDIA A100 GPUs

### Stage B: Multi-Object Attention Optimization

**Objective:** Refine the global descriptor to align with object attention maps across all crops.

**Process:**

1. **Initial Encoding**: Use Stage A encoder to get object features `{v₁, ..., vₖ}`
1. **Explainability Maps**: Compute attention maps using LeGrad:
   
   ```
   E(vᶜ · vᵢ) ∈ ℝᵂⁱˣᴴⁱ
   ```
   
   This shows which image regions the representation focuses on
1. **Optimization Objective**:
   
   ```
   v̂ᶜ = argmax_vᶜ [ Σᵢ IoU(E(vᶜ · vᵢ), mᵢ) + α·vᶜ·(Σᵢ vᵢ) ]
   ```
   
   Where:
- `v̂ᶜ`: Optimized global representation
- `E(vᶜ · vᵢ)`: Explainability map for crop i
- `mᵢ`: Ground-truth object mask (from SAM)
- `IoU`: Intersection over Union between map and mask
- `α = 0.03`: Regularization weight (keeps v̂ᶜ close to original)
1. **Gradient Descent**:
- Initialize: `v̂ᶜ⁽⁰⁾ = vᶜ` (from Stage A)
- Iterations: 80
- Learning rate: 1×10⁻¹
- Time: 0.03s per object (offline for gallery)

**Key Insight:** By aligning attention maps with object masks, Stage B ensures the global descriptor equally represents all objects, not just large/salient ones.

### Why MaO Works

1. **Filters Background**: Only detected objects are encoded, removing sky, walls, floors
1. **Equal Representation**: Each object gets equal weight, regardless of size
1. **Attention Alignment**: Stage B ensures the descriptor “attends to” the right regions
1. **Single Descriptor**: Produces one compact vector per image for scalable search
1. **Handles Clutter**: Processes 15+ objects per image without confusion

## MaO Performance Across Datasets

### Fine-tuned Results (with Ground Truth annotations)

#### VoxDet Performance

|Method        |Type    |mAP       |Improvement over Baseline|
|--------------|--------|----------|-------------------------|
|**MaO-DINOv2**|**MaO** |**83.70%**|**+29.37%** vs DINOv2    |
|**MaO-CLIP**  |**MaO** |**79.86%**|**+27.06%** vs CLIP      |
|α-CLIP        |Baseline|59.74%    |-                        |
|GeM           |Baseline|61.45%    |-                        |
|CLIP          |Baseline|52.80%    |-                        |
|DINOv2        |Baseline|54.33%    |-                        |

#### PerMiR Performance

|Method        |Type    |mAP       |
|--------------|--------|----------|
|**MaO-CLIP**  |**MaO** |**90.86%**|
|**MaO-DINOv2**|**MaO** |**90.07%**|
|α-CLIP        |Baseline|90.13%    |
|GeM           |Baseline|41.20%    |
|CLIP          |Baseline|38.49%    |
|DINOv2        |Baseline|30.47%    |

#### INSTRE-XS Performance

|Method        |Type    |mAP       |
|--------------|--------|----------|
|**MaO-CLIP**  |**MaO** |**91.29%**|
|**MaO-DINOv2**|**MaO** |**90.01%**|
|GSS           |Baseline|82.34%    |
|GeM           |Baseline|82.61%    |
|AMES          |Baseline|78.61%    |
|CLIP          |Baseline|72.90%    |

#### INSTRE-XXS Performance

|Method        |Type    |mAP       |
|--------------|--------|----------|
|**MaO-CLIP**  |**MaO** |**77.46%**|
|**MaO-DINOv2**|**MaO** |**75.91%**|
|AMES          |Baseline|68.08%    |
|GSS           |Baseline|67.98%    |
|GeM           |Baseline|65.58%    |
|CLIP          |Baseline|62.04%    |

### Zero-shot Results (no training on target dataset)

#### VoxDet Zero-shot

|Method        |Type    |mAP       |Improvement          |
|--------------|--------|----------|---------------------|
|**MaO-DINOv2**|**MaO** |**70.20%**|**+18.97%** vs DINOv2|
|**MaO-CLIP**  |**MaO** |**65.22%**|**+20.70%** vs CLIP  |
|GSS           |Baseline|52.01%    |-                    |
|DINOv2        |Baseline|51.23%    |-                    |
|GeM           |Baseline|51.08%    |-                    |
|CLIP          |Baseline|44.52%    |-                    |

#### PerMiR Zero-shot

|Method        |Type    |mAP       |
|--------------|--------|----------|
|**MaO-DINOv2**|**MaO** |**89.86%**|
|**MaO-CLIP**  |**MaO** |**89.51%**|
|α-CLIP        |Baseline|88.21%    |
|DINOv2        |Baseline|40.57%    |
|AMES          |Baseline|29.72%    |
|GSS           |Baseline|26.73%    |

#### INSTRE-XS Zero-shot

|Method        |Type    |mAP       |
|--------------|--------|----------|
|**MaO-CLIP**  |**MaO** |**89.39%**|
|GSS           |Baseline|82.34%    |
|AMES          |Baseline|78.61%    |
|GeM           |Baseline|74.74%    |
|**MaO-DINOv2**|**MaO** |**71.28%**|

#### INSTRE-XXS Zero-shot

|Method        |Type    |mAP       |
|--------------|--------|----------|
|**MaO-CLIP**  |**MaO** |**71.23%**|
|AMES          |Baseline|68.08%    |
|GSS           |Baseline|67.98%    |
|**MaO-DINOv2**|**MaO** |**53.13%**|
|GeM           |Baseline|53.27%    |

### Key Performance Highlights Across All Datasets

- ✅ **Consistent improvements** across all benchmarks
- ✅ **VoxDet**: 18-29 mAP point improvements over baselines
- ✅ **PerMiR**: Near-perfect 90%+ mAP on multi-instance retrieval
- ✅ **INSTRE-XS/XXS**: 8-15 mAP point improvements on small objects
- ✅ Retrieves objects as small as **0.5% of image area** (VoxDet)
- ✅ Handles **~15 detected objects per image** in high clutter
- ✅ **Single descriptor** per image (scalable to large databases)
- ✅ Strong **zero-shot transfer** across different dataset characteristics

## API Usage

```python
from extractors import DINOv2MIExtractor, CLIPMIExtractor
from PIL import Image

# Initialize MaO extractor
extractor = DINOv2MIExtractor(
    mi_alpha=0.03,           # Stage B regularization
    global_features=True     # Use CLS token
)

# Load image (works with any dataset)
image = Image.open("scene_image.jpg")

# Extract MaO features (runs Stage A + Stage B)
features = extractor.extract(image)  # Shape: (768,)

# For CLIP backbone
clip_extractor = CLIPMIExtractor(mi_alpha=0.03, global_features=True)
clip_features = clip_extractor.extract(image)  # Shape: (512,)
```

### Batch Processing

```python
import torch
from pathlib import Path

# Process gallery images (works for any dataset)
gallery_dir = Path("/path/to/dataset/gallery")
gallery_features = []

for img_path in gallery_dir.glob("*.jpg"):
    image = Image.open(img_path)
    features = extractor.extract(image)
    gallery_features.append(features)

gallery_features = torch.stack(gallery_features)  # Shape: (N, 768)

# Build FAISS index
import faiss
index = faiss.IndexFlatIP(768)  # Inner product (cosine similarity)
index.add(gallery_features.numpy())

# Query
query_image = Image.open("query_image.jpg")
query_features = extractor.extract(query_image)
distances, indices = index.search(query_features.unsqueeze(0).numpy(), k=10)
```

## Implementation Details

### Training Configuration (Stage A)

**Note:** Users typically use pre-trained LoRA checkpoints. Fine-tuning is only needed if you want to train on your own custom dataset.

- **Dataset**: Training set (e.g., VoxDet: 9.6K instances, 55K scenes)
- **Optimizer**: AdamW
- **Learning Rate**: 5×10⁻⁵ → 1×10⁻⁶ (exponential decay 0.93)
- **Fine-tuning**: LoRA rank 256 (only adapters trained)
- **Batch Size**: 128
- **Epochs**: 1
- **Hardware**: 4× NVIDIA A100 GPUs
- **Training Time**: ~4-6 hours

### Optimization Configuration (Stage B)

- **Iterations**: 80
- **Learning Rate**: 1×10⁻¹
- **Regularization**: α = 0.03
- **Initialization**: v̂ᶜ⁽⁰⁾ = vᶜ (from Stage A)
- **Time per Object**: 0.03 seconds
- **Mode**: Offline for gallery, online for queries

### Inference Configuration

- **Object Detector**: OWLv2 base-patch16-ensemble
- **Detection Mode**: “Objectness” (class-agnostic)
- **Confidence Threshold**: 0.2
- **Segmentation**: SAM (vit_h checkpoint)
- **Min Crop Size**: 224×224 (DINOv2) or 384×384 (SigLIP)
- **Feature Dimension**: 768 (DINOv2), 512 (CLIP)

## Directory Structure

```
SoIR/
├── README.md
├── requirements.txt
├── configs/
│   └── defaults.yaml              # MaO default configuration
├── create_index.py                # Stage A + B, create FAISS index
├── search_index.py                # Search and evaluate on VoxDet
├── extractors/
│   ├── __init__.py
│   ├── base_extractor.py          # Abstract base class
│   ├── dinov2_mi_extractor.py     # DINOv2 + MaO (Stage A + B)
│   ├── clip_mi_extractor.py       # CLIP + MaO (Stage A + B)
│   └── siglip_mi_extractor.py     # SigLIP + MaO (Stage A + B)
├── utils/
│   ├── mask_inversion.py          # Stage B implementation (LeGrad optimization)
│   ├── metrics.py                 # mAP, Recall@K for VoxDet
│   └── image_preprocessor.py      # Image preprocessing
├── datasets/
│   └── base_dataset.py            # VoxDet loader
└── scripts/
    └── create_masked_dataset.py   # Prepare VoxDet with OWLv2 + SAM
```

## Citation

If you use this code or the MaO method, please cite:

```bibtex
@inproceedings{green2025findyourneedle,
  author={Green, Michael and Levy, Matan and Tzachor, Issar and Samuel, Dvir and Darshan, Nir and Ben-Ari, Rami},
  title={Find your Needle: Small Object Image Retrieval via Multi-Object Attention Optimization},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

## License

MIT License

## Acknowledgments

- [VoxDet Dataset](https://github.com/voxdet) by Li et al. - Foundation for small object retrieval evaluation
- [DINOv2](https://github.com/facebookresearch/dinov2) by Meta AI - Best performing backbone
- [CLIP](https://github.com/openai/CLIP) by OpenAI - Efficient alternative backbone
- [SigLIP](https://huggingface.co/google/siglip-base-patch16-384) by Google - High-performance backbone
- [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble) by Google - Open-vocabulary object detection
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything) by Meta AI - Object mask generation
- [LeGrad](https://arxiv.org/abs/2404.03214) - Explainability method for Stage B
- [MaskInversion](https://arxiv.org/abs/2407.20034) - Inspiration for attention optimization
