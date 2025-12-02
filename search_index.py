"""
Main script for searching a FAISS index.

Usage:
    python search_index.py dataset=your_dataset experiment=experiment_name features_dir=/path/to/features [options]

Example:
    python search_index.py \
        dataset=custom \
        experiment=custom_dinov2_mi_features_global_smart_crop \
        k_search=100 \
        features_dir=/path/to/outputs \
        save_knns=true
"""

import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from tqdm import tqdm
from PIL import Image

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
    USE_FAISS = False

# Local imports
from utils.metrics import (
    get_recalls,
    calculate_map,
    calculate_binary_metrics,
    find_optimal_threshold,
    analyze_label_distribution
)


def get_logger(log_path):
    """Setup logger."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def keep_first_occurrence(tensor):
    """Keep only first occurrence of each unique value."""
    flat_tensor = tensor.reshape(-1)
    unique_vals, indices = torch.unique(flat_tensor, return_inverse=True)
    
    first_occurrences = torch.zeros_like(indices, dtype=torch.bool)
    for i in range(len(unique_vals)):
        first_idx = torch.where(indices == i)[0][0]
        first_occurrences[first_idx] = True
    
    return flat_tensor[first_occurrences]


def search_index(keypoints_file, out_file, args, logger=None):
    """
    Search the FAISS index with query embeddings.
    
    Args:
        keypoints_file: Path to the keypoints file
        out_file: Output file for results
        args: Configuration arguments
        logger: Logger instance
        
    Returns:
        Results dictionary
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = {}
    keypoints_file = Path(keypoints_file)
    save_knns = getattr(args, 'save_knns', False)
    
    # Setup paths
    keypoints_query_dir = keypoints_file.with_name(
        keypoints_file.stem + "_kpts_query"
    )
    faiss_index_file = keypoints_file.with_name(
        keypoints_file.stem + "_gallery.bin"
    )
    
    logger.info(f"Loading keypoints from {keypoints_query_dir}")
    logger.info(f"Loading FAISS index from {faiss_index_file}")
    
    # Load FAISS index
    if USE_FAISS:
        faiss_index = faiss.read_index(str(faiss_index_file))
    else:
        raise ImportError("FAISS is required for searching. Please install faiss-gpu or faiss-cpu.")
    
    # Load query files
    query_files = sorted(list(keypoints_query_dir.iterdir()))
    
    # Determine search parameters
    k_search = min(faiss_index.ntotal, int(getattr(args, 'max_k_search', 1000000)))
    ranks = np.concatenate((np.array([1]), np.arange(start=5, stop=k_search + 1, step=5)))
    
    # Load first and last query files for metadata
    first_query = torch.load(query_files[0])
    last_query = torch.load(query_files[-1]) if len(query_files) > 1 else first_query
    
    gallery_labels = last_query["gallery"]["labels"]
    gallery_paths = last_query["gallery"].get("paths", None)
    sc_index = last_query["gallery"].get("sc_index", None)
    
    # Check for multi-label
    multi_label = False
    if isinstance(gallery_labels[0], list):
        max_num_lbls = max(len(gl) for gl in gallery_labels)
        multi_label = max_num_lbls > 1
    else:
        max_num_lbls = 1
    
    LIMIT = first_query["LIMIT"]["MAX"]
    
    # Calculate total queries
    if len(query_files) > 1:
        len_q = (LIMIT * (len(query_files) - 1)) + len(last_query["queries"])
    else:
        len_q = len(last_query["queries"])
    
    # Initialize tracking variables
    cnt_q = 0
    cur_file = query_files[cnt_q]
    queries = first_query["queries"]
    
    all_q_paths = []
    all_g_paths = []
    labels_knn = []
    labels_q = []
    query_paths = []
    times = []
    knns = {}
    sc_index_dict = {}
    all_similarities = []
    all_k_indices = []
    
    q_ind = 0
    
    logger.info(f"Processing {len_q} queries...")
    
    while q_ind < len_q:
        print(f"Processing queries {q_ind}/{len_q}")
        
        # Check if we need to load next file
        file_ind = q_ind // LIMIT + 1
        if file_ind > int(cur_file.stem):
            cnt_q += 1
            cur_file = query_files[cnt_q]
            queries = torch.load(cur_file)["queries"]
            query_paths = []
        
        start_time = time.time()
        all_q_paths += list(queries.keys())
        
        # Get query embeddings
        q_embeddings = [
            queries[q_path]["pts"]["keypoints"] for q_path in queries
        ]
        q_embeddings = torch.cat(q_embeddings)
        q_embeddings = torch.nn.functional.normalize(q_embeddings, p=2, dim=-1)
        
        # Search
        similarities, k_index = faiss_index.search(
            q_embeddings.cpu().data.numpy(), k_search
        )
        k_index = torch.from_numpy(k_index)
        
        all_similarities.append(similarities)
        all_k_indices.append(k_index.cpu().numpy())
        
        # Get query labels
        _lbls_q = []
        for q_path in queries:
            num_vecs = queries[q_path]["pts"]["keypoints"].shape[0]
            _lbls_q += queries[q_path]["label"] * num_vecs
            query_paths += [q_path] * num_vecs
        labels_q += _lbls_q
        
        # Get KNN labels
        for knn_ind in range(len(k_index)):
            try:
                if not multi_label:
                    _g_lbls = np.array(gallery_labels)[k_index[knn_ind].numpy()]
                else:
                    _g_lbls = -1 * np.ones((len(k_index[knn_ind]), max_num_lbls))
                    for cnt_lbl, gl_ind in enumerate(k_index[knn_ind].numpy()):
                        gl = gallery_labels[gl_ind]
                        _g_lbls[cnt_lbl][:len(gl)] = gl
                labels_knn.append(_g_lbls)
            except Exception as e:
                labels_knn.append(np.zeros(k_index[knn_ind].shape))
        
        # Save KNNs if requested
        if save_knns:
            for _q_ind in range(len(k_index)):
                q_path = query_paths[_q_ind] if _q_ind < len(query_paths) else list(queries.keys())[_q_ind]
                # Filter out indices that are out of range
                valid_indices = [int(_ind) for _ind in k_index[_q_ind] if _ind < len(gallery_paths)]
                knns[q_path] = [gallery_paths[_ind] for _ind in valid_indices]
                if sc_index:
                    sc_index_dict[q_path] = [sc_index[_ind] for _ind in valid_indices if _ind < len(sc_index)]
        
        q_ind += LIMIT
        times.append(time.time() - start_time)
    
    # Combine all results
    all_similarities = np.vstack(all_similarities) if all_similarities else np.array([])
    all_k_indices = np.vstack(all_k_indices) if all_k_indices else np.array([])
    
    # Analyze label distribution
    label_stats = analyze_label_distribution(gallery_labels, labels_q)
    logger.info(f"Gallery: {label_stats['gallery_stats']['avg_labels_per_item']:.2f} labels/item")
    logger.info(f"Queries: {label_stats['query_stats']['avg_labels_per_item']:.2f} labels/item")
    
    # Calculate binary metrics
    logger.info("Calculating binary classification metrics...")
    binary_metrics = calculate_binary_metrics(
        similarities=all_similarities,
        k_index=all_k_indices,
        gallery_labels=gallery_labels,
        query_labels=labels_q,
        thresholds=np.arange(0.1, 1.0, 0.05),
        multi_label=multi_label,
        max_num_lbls=max_num_lbls
    )
    
    # Find optimal threshold
    optimal_result = find_optimal_threshold(binary_metrics, criterion='f1')
    logger.info(f"Optimal threshold (F1): {optimal_result['threshold']:.3f}")
    logger.info(f"Precision: {optimal_result['precision']:.3f}, Recall: {optimal_result['recall']:.3f}, F1: {optimal_result['f1_score']:.3f}")
    logger.info(f"Average Precision: {binary_metrics['average_precision']:.3f}")
    logger.info(f"AUC: {binary_metrics['auc']:.3f}")
    
    # Calculate recalls
    recall, hits = get_recalls(labels_knn, labels_q, ranks)
    
    # Calculate mAP
    mAP, prec_k = calculate_map(np.array(labels_knn), labels_q, k=3)
    
    # Build results dictionary
    results["binary_metrics"] = binary_metrics
    results["similarities"] = all_similarities
    results["recall"] = recall
    results["hits"] = hits
    results["knns"] = knns
    if sc_index_dict:
        results["sc_index"] = sc_index_dict
    results["labels_knn"] = labels_knn
    results["query_labels"] = labels_q
    results["gallery_labels"] = gallery_labels
    results["queries"] = all_q_paths
    results["gallery"] = all_g_paths
    results["kpts_file"] = str(keypoints_file)
    results["mAP"] = mAP
    results["precision@k"] = prec_k
    results["avg_time"] = np.round(
        100 / len(gallery_labels) * np.mean(times), 2
    )
    
    # Log results
    logger.info(f"Results saved to {out_file}")
    recall_string = pformat(recall)
    logger.info(recall_string)
    
    recall_values = [str(round(x, 4)) for x in list(recall.values())[:10]]
    logger.info("Recalls: " + ", ".join(recall_values))
    logger.info(f"mAP: {mAP:.4f}")
    logger.info(f"Avg runtime per 100 queries: {results['avg_time']} seconds")
    
    # Save results
    torch.save(results, str(out_file))
    
    return results


def get_keypoints_file(args):
    """Get keypoints file path from configuration."""
    features_dir = getattr(args, 'features_dir', '.')
    experiment = getattr(args, 'experiment', '')
    dataset = getattr(args, 'dataset', 'custom')
    
    keypoints_file = Path(features_dir) / f"{dataset}_{experiment}.pt"
    
    return keypoints_file


@hydra.main(config_path="configs", config_name="defaults", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point."""
    OmegaConf.set_struct(cfg, False)
    args = cfg
    
    # Setup paths
    features_dir = getattr(args, 'features_dir', '.')
    experiment = getattr(args, 'experiment', '')
    dataset = getattr(args, 'dataset', 'custom')
    
    # Construct keypoints file path
    keypoints_file = Path(features_dir) / f"{experiment}.pt"
    
    # Check alternative paths
    if not keypoints_file.exists():
        # Try without .pt suffix (for directory-based storage)
        keypoints_file = Path(features_dir) / experiment
    
    # Output file
    out_file = Path(f"{dataset}_{experiment}_results.pt")
    
    # Setup logger
    log_path = Path(f"{dataset}_{experiment}.log")
    logger = get_logger(str(log_path))
    
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Features dir: {features_dir}")
    logger.info(f"Keypoints file: {keypoints_file}")
    
    # Run search
    search_index(
        keypoints_file=keypoints_file,
        out_file=out_file,
        args=args,
        logger=logger
    )


if __name__ == "__main__":
    main()
