"""
Main script for creating a FAISS index from gallery images.

Usage:
    python create_index.py dataset=your_dataset anns_file=/path/to/anns.pt [options]

Example:
    python create_index.py \
        dataset=custom \
        extractor=dinov2_mi_features \
        B_model=true \
        vec_dim=768 \
        mi_alpha=0.03 \
        weights=/path/to/weights.ckpt \
        lora_adapt=true \
        lora_rank=256 \
        smart_crop=true \
        global_features=true \
        captions_file=/path/to/captions.pt \
        anns_file=/path/to/annotations.pt
"""

import logging
import os
import sys

# Add the package directory to path for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from datetime import datetime

# Hydra for configuration
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

# FAISS for similarity search
try:
    import faiss
    import faiss.contrib.torch_utils
    USE_FAISS = True
except ImportError:
    print("FAISS not found. Using sklearn-based similarity search.")
    USE_FAISS = False

# Local imports
from utils.image_preprocessor import ImagePreprocessor, resize_by_scale
from extractors.dinov2_extractor import DinoV2Extractor
from extractors.dinov2_mi_extractor import DinoV2MIExtractor
from datasets.base_dataset import BaseDataset, get_dataset_class


LIMIT = 1000  # Queries per file


def get_extractor(args):
    """
    Get the feature extractor based on configuration.
    
    Args:
        args: Configuration object
        
    Returns:
        extractor: Feature extractor instance
        ds_class: Dataset class
        ds_kwargs: Dataset keyword arguments
        out_file: Output file path
    """
    extractor_name = getattr(args, 'extractor', 'dinov2_features')
    
    # Create extractor
    if 'clip_mi_features' in extractor_name:
        from extractors.clip_mi_extractor import CLIPMIExtractor
        extractor = CLIPMIExtractor(args=args)
    elif 'clip_features' in extractor_name:
        from extractors.clip_extractor import CLIPExtractor
        extractor = CLIPExtractor(args=args)
    elif 'siglip_mi_features' in extractor_name:
        from extractors.siglip_mi_extractor import SigLIPMIExtractor
        extractor = SigLIPMIExtractor(args=args)
    elif 'siglip_features' in extractor_name:
        from extractors.siglip_extractor import SigLIPExtractor
        extractor = SigLIPExtractor(args=args)
    elif 'dinov2_mi_features' in extractor_name:
        extractor = DinoV2MIExtractor(args=args)
    elif 'dinov2_features' in extractor_name:
        extractor = DinoV2Extractor(args=args)
    else:
        raise ValueError(f"Unknown extractor: {extractor_name}")
    
    # Get dataset info
    ds_class, ds_kwargs = get_dataset_class(args)
    
    # Build output file path
    add_string = f"_{extractor_name}"
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
    if getattr(args, 'weights', None):
        add_string += '_FT_weights'
    
    dataset = getattr(args, 'dataset', 'custom')
    out_file = f"{dataset}{add_string}.pt"
    
    return extractor, ds_class, ds_kwargs, out_file


def add_to_dict(out_dict, keypoints, label, index, ds, bbox, img_path, return_dict=False):
    """Add extracted features to the output dictionary."""
    _dict = {}
    
    if isinstance(keypoints, tuple):
        keypoints = keypoints[0]
    
    _dict["pts"] = keypoints
    
    if not isinstance(label, list):
        _dict["label"] = [label]
    else:
        _dict["label"] = label
    
    cls_name = ds.cls_names[index]
    _dict["cls_name"] = cls_name
    _dict["bbox"] = bbox
    
    if not return_dict:
        if img_path in out_dict["queries"]:
            _dict["label"] += out_dict["queries"][img_path]["label"]
            _dict["label"] = np.unique(_dict["label"]).tolist()
        out_dict["queries"][img_path] = _dict
        return None
    else:
        _dict["img_path"] = img_path
        return _dict


def save_embeddings(embeddings, labels, paths, save_path, chunk_index):
    """Save embeddings to disk."""
    out_path = save_path / f"gall_kpts_chunk_{chunk_index:02d}.pt"
    torch.save(
        {"embeddings": embeddings, "labels": labels, "paths": paths},
        out_path
    )
    logging.info(f"Saved embeddings chunk {chunk_index} at {out_path}")


def create_faiss_index(
    ds_class,
    out_file,
    ds_kwargs,
    extractor,
    pre_processor_conf=None,
    args=None
):
    """
    Create FAISS index from gallery images.
    
    Args:
        ds_class: Dataset class
        out_file: Output file path
        ds_kwargs: Dataset keyword arguments
        extractor: Feature extractor
        pre_processor_conf: Image preprocessor configuration
        args: Additional arguments
    """
    # Determine vector dimension
    vec_dim = getattr(args, 'vec_dim', 768)
    gallery_chunk_size = getattr(args, 'gallery_chunk_size', 10000)
    
    # Initialize FAISS index
    if USE_FAISS and not getattr(args, 'no_faiss', False):
        faiss_index = faiss.IndexFlatIP(vec_dim)
    else:
        faiss_index = None
    
    # Setup paths
    out_file = Path(out_file)
    gallery_kpts_path = out_file.with_name(out_file.stem + "_gallery.bin")
    query_kpts_path = out_file.with_name(out_file.stem + "_kpts_query")
    gall_kpts_path = out_file.with_name(out_file.stem + "_kpts_gall")
    
    query_kpts_path.mkdir(exist_ok=True, parents=True)
    gall_kpts_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize dataset
    ds = ds_class(**ds_kwargs)
    
    # Setup image preprocessor
    format_type = 'XYXY'
    if pre_processor_conf is None:
        pre_processor_conf = {}
    
    image_pre_processor = ImagePreprocessor(
        format=format_type,
        crop_no_resize=getattr(args, 'crop', False),
        args=args,
        **pre_processor_conf
    )
    
    # Initialize output structures
    outDict = defaultdict(dict)
    gallery_labels = []
    gallery_paths = []
    embeddings_list = []
    chunk_index = 0
    sc_index = []
    
    cnt_q = 0
    cnt_q_save = 0
    
    outDict["gallery"]["dir"] = str(gallery_kpts_path)
    
    # Process dataset
    logging.info(f"Processing {len(ds)} images...")
    
    for item in tqdm(ds):
        img_pil, img_path, (index, bbox, label, scale) = item
        is_query = ds.is_query(index)
        
        # Skip gallery if only_gallery is False
        if is_query and getattr(args, 'only_gallery', False):
            continue
        
        # Check max index
        max_ds_index = getattr(args, 'max_ds_index', None)
        if max_ds_index is not None and index > max_ds_index:
            break
        
        # Handle masks
        masks = None
        if isinstance(bbox, dict):
            masks = bbox.get('masks', None)
            bbox = bbox.get('bbox', None)
        
        # Preprocess bbox
        if bbox is not None:
            if isinstance(bbox, np.ndarray) and bbox.ndim > 1:
                bbox = bbox[0]
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], (list, tuple)):
                    bbox = bbox[0]
        
        # Preprocess image
        is_query_pre = is_query if not getattr(args, 'no_query_crop', False) else False
        q_im = image_pre_processor(
            img_pil,
            bbox=bbox,
            is_query=is_query_pre,
            from_detector=False
        )
        
        # Resize if needed
        resize_scale = getattr(args, 'resize_scale', None)
        if resize_scale is not None:
            q_im = resize_by_scale(q_im, resize_scale)
        
        # Prepare input for extractor
        add_name_to_extractor = getattr(args, 'captions_file', None) is not None
        if add_name_to_extractor and masks is None:
            q_im = {"image": q_im, "img_path": img_path}
        
        # Extract features
        extractor.is_current_query = is_query
        keypoints = extractor.extract(q_im, is_query=is_query, masks=masks)
        
        # Handle multi-vector output
        if keypoints["keypoints"].ndim > 2 and is_query:
            keypoints["keypoints"] = torch.squeeze(keypoints["keypoints"])
        
        # Average multiple vectors if configured
        if getattr(args, 'avg_multiple_vecs', False):
            if keypoints["keypoints"].ndim > 2:
                keypoints["keypoints"] = torch.squeeze(keypoints["keypoints"])
                if keypoints["keypoints"].ndim == 1:
                    keypoints["keypoints"] = keypoints["keypoints"][None]
            
            if hasattr(extractor, 'merge_multiple_vecs'):
                keypoints = extractor.merge_multiple_vecs(
                    keypoints, key="keypoints", img_path=img_path
                )
                if not is_query:
                    keypoints["keypoints"] = keypoints["keypoints"][:, None]
            else:
                keypoints["keypoints"] = torch.sum(
                    keypoints["keypoints"], dim=0, keepdim=True
                )
        
        # Add to output dictionary
        return_dict = not is_query
        _dict = add_to_dict(
            out_dict=outDict,
            keypoints=keypoints,
            label=label,
            index=index,
            ds=ds,
            bbox=bbox,
            img_path=img_path,
            return_dict=return_dict
        )
        
        # Handle gallery items
        if not is_query:
            embeddings = keypoints["keypoints"]
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
            if faiss_index is not None:
                # Handle different embedding dimensions
                if embeddings.ndim > 2:
                    embeddings = torch.squeeze(embeddings)
                if embeddings.ndim == 1:
                    embeddings = embeddings[None]
                
                # Number of embeddings to add (may be multiple per image with smart_crop)
                num_embeddings = embeddings.shape[0]
                
                faiss_index.add(embeddings.cpu().data)
                sc_index += list(range(num_embeddings))
                
                # Add a label/path for each embedding
                for _ in range(num_embeddings):
                    gallery_labels.append(label)
                    gallery_paths.append(img_path)
            else:
                embeddings_list.append(embeddings)
                gallery_labels.append(label)
                gallery_paths.append(img_path)
                
                if len(embeddings_list) >= gallery_chunk_size:
                    save_embeddings(
                        embeddings_list, gallery_labels, gallery_paths,
                        gall_kpts_path, chunk_index
                    )
                    embeddings_list = []
                    gallery_labels = []
                    gallery_paths = []
                    chunk_index += 1
        else:
            cnt_q += 1
        
        # Save queries periodically
        if cnt_q >= LIMIT:
            cnt_q_save += 1
            cnt_q = 0
            
            if cnt_q_save == 1:
                outDict["LIMIT"]["MAX"] = LIMIT
            
            outDict["gallery"]["labels"] = gallery_labels
            outDict["gallery"]["paths"] = gallery_paths
            outDict["gallery"]["sc_index"] = sc_index
            
            _out_file_q = query_kpts_path / f"{cnt_q_save}.pt"
            torch.save(outDict, _out_file_q)
            logging.info(f"Saved query dict to {_out_file_q}")
            outDict = defaultdict(dict)
    
    # Save remaining queries
    if cnt_q > 0:
        if cnt_q_save == 0:
            outDict["LIMIT"]["MAX"] = LIMIT
        
        cnt_q_save += 1
        _out_file_q = query_kpts_path / f"{cnt_q_save}.pt"
        outDict["gallery"]["labels"] = gallery_labels
        outDict["gallery"]["paths"] = gallery_paths
        outDict["gallery"]["sc_index"] = sc_index
        torch.save(outDict, _out_file_q)
        logging.info(f"Saved query dict to {_out_file_q}")
    
    # Save FAISS index
    if faiss_index is not None:
        if USE_FAISS:
            faiss.write_index(faiss_index, str(gallery_kpts_path))
        logging.info(f"Saved FAISS index to {gallery_kpts_path}")
    else:
        # Save remaining embeddings
        if embeddings_list:
            save_embeddings(
                embeddings_list, gallery_labels, gallery_paths,
                gall_kpts_path, chunk_index
            )
    
    logging.info("Index creation complete!")
    return str(out_file)


@hydra.main(config_path="configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    logging.info(f"Command: python {os.path.basename(__file__)} {' '.join(sys.argv[1:])}")
    
    # Allow modification of config
    OmegaConf.set_struct(cfg, False)
    args = cfg
    
    # Get output directory from Hydra
    output_dir = HydraConfig.get().runtime.output_dir
    
    # Setup preprocessor config
    pre_processor_conf = None
    if getattr(args, 'blur', -1) != -1:
        from PIL import ImageFilter
        blur_func = ImageFilter.GaussianBlur(args.blur)
        pre_processor_conf = {"blur": blur_func}
    
    if getattr(args, 'deblur', -1) != -1:
        from PIL import ImageFilter
        if pre_processor_conf is None:
            pre_processor_conf = {}
        pre_processor_conf["deblur"] = ImageFilter.SHARPEN
        pre_processor_conf["deblur_num"] = args.deblur
    
    # Get extractor and dataset
    extractor, ds_class, ds_kwargs, out_file = get_extractor(args)
    
    args.out_file = out_file
    args.features_dir = output_dir
    
    # Create index
    logging.info("Creating FAISS index...")
    out_file = create_faiss_index(
        ds_class=ds_class,
        out_file=out_file,
        ds_kwargs=ds_kwargs,
        pre_processor_conf=pre_processor_conf,
        extractor=extractor,
        args=args
    )
    
    # Run search if not only_create
    if not getattr(args, 'only_create', False):
        logging.info("Running search...")
        
        # Get the directory containing this script
        script_dir = Path(__file__).parent.resolve()
        search_script = script_dir / "search_index.py"
        
        # Build search command
        cmd = f"python {search_script}"
        cmd += f" dataset={args.dataset}"
        cmd += f" experiment={Path(out_file).stem}"
        cmd += f" k_search={getattr(args, 'k_search', 100)}"
        cmd += f" features_dir={output_dir}"
        
        if getattr(args, 'save_knns', False):
            cmd += " save_knns=true"
        
        logging.info(f"Running: {cmd}")
        os.system(cmd)


if __name__ == "__main__":
    main()
