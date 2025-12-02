"""
Evaluation metrics for retrieval systems.
"""

import numpy as np
import torch
from collections import defaultdict


def get_recalls(labels_g, labels_q, ranks):
    """
    Calculate recall at different ranks.
    
    Args:
        labels_g: Gallery labels for retrieved items [n_queries, k]
        labels_q: Query labels [n_queries]
        ranks: List of ranks to compute recall at
        
    Returns:
        recall: Dict mapping rank to recall value
        hits: Dict mapping rank to per-query hit indicators
    """
    labels_g = np.array(labels_g)
    labels_q = np.array(labels_q)
    
    hits = {}
    recall = {}
    
    for rank in ranks:
        lbls_g = labels_g[:, :rank]
        _hits = []
        for ind in range(len(labels_q)):
            _hit = labels_q[ind] in lbls_g[ind]
            _hits.append(_hit)
        hits[rank] = _hits
        recall[rank] = np.mean(_hits)
    
    return recall, hits


def calculate_map(scores, labels_q, knns=None, k=None):
    """
    Calculate mean Average Precision (mAP).
    
    Args:
        scores: Retrieved item labels or score matrix
        labels_q: Query labels
        knns: Dict mapping query paths to gallery paths (for deduplication)
        k: Top-k for precision calculation
        
    Returns:
        mAP: Mean average precision
        precision_at_k: Precision at k
    """
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    scores = np.array(scores)
    labels_q = np.array(labels_q)
    
    n_queries = len(labels_q)
    aps = []
    precs_at_k = []
    
    for i in range(n_queries):
        query_label = labels_q[i]
        retrieved_labels = scores[i]
        
        # Handle multi-label case
        if isinstance(query_label, (list, np.ndarray)):
            query_labels_set = set(query_label) if hasattr(query_label, '__iter__') else {query_label}
        else:
            query_labels_set = {query_label}
        
        # Calculate relevance
        if retrieved_labels.ndim == 1:
            relevance = np.array([l in query_labels_set for l in retrieved_labels])
        else:
            # Multi-label retrieved items
            relevance = np.array([
                bool(set(l[l >= 0]) & query_labels_set) for l in retrieved_labels
            ])
        
        # Calculate AP
        if relevance.sum() == 0:
            aps.append(0.0)
        else:
            precisions = []
            n_relevant = 0
            for j, rel in enumerate(relevance):
                if rel:
                    n_relevant += 1
                    precisions.append(n_relevant / (j + 1))
            aps.append(np.mean(precisions) if precisions else 0.0)
        
        # Calculate precision at k
        if k is not None and k <= len(relevance):
            precs_at_k.append(relevance[:k].mean())
    
    mAP = np.mean(aps)
    precision_at_k = np.mean(precs_at_k) if precs_at_k else 0.0
    
    return mAP, precision_at_k


def calculate_binary_metrics(
    similarities, 
    k_index, 
    gallery_labels, 
    query_labels,
    thresholds=None, 
    multi_label=False, 
    max_num_lbls=1
):
    """
    Calculate binary classification metrics at different thresholds.
    
    Args:
        similarities: Similarity scores [n_queries, k]
        k_index: Retrieved item indices [n_queries, k]
        gallery_labels: Labels for gallery items
        query_labels: Labels for query items
        thresholds: Thresholds to evaluate
        multi_label: Whether items can have multiple labels
        max_num_lbls: Maximum number of labels per item
        
    Returns:
        Dict with precision, recall, F1, AUC, etc.
    """
    if isinstance(similarities, torch.Tensor):
        similarities = similarities.cpu().numpy()
    if isinstance(k_index, torch.Tensor):
        k_index = k_index.cpu().numpy()
    
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    thresholds = np.array(thresholds)
    
    n_queries = len(query_labels)
    n_thresholds = len(thresholds)
    
    precisions = np.zeros(n_thresholds)
    recalls = np.zeros(n_thresholds)
    f1_scores = np.zeros(n_thresholds)
    accuracies = np.zeros(n_thresholds)
    
    def labels_match(label1, label2):
        """Check if two labels match."""
        if not isinstance(label1, (list, np.ndarray)):
            label1 = [label1]
        if not isinstance(label2, (list, np.ndarray)):
            label2 = [label2]
        return bool(set(label1) & set(label2))
    
    # Compute total positives for each query
    total_positives = np.zeros(n_queries)
    for q_idx, query_label in enumerate(query_labels):
        for g_idx, gallery_label in enumerate(gallery_labels):
            if labels_match(query_label, gallery_label):
                total_positives[q_idx] += 1
    
    for t_idx, threshold in enumerate(thresholds):
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_tn = 0
        
        for q_idx in range(n_queries):
            query_label = query_labels[q_idx]
            retrieved_indices = k_index[q_idx]
            retrieved_sims = similarities[q_idx]
            
            # Count TP and FP above threshold
            tp = 0
            fp = 0
            for sim, ret_idx in zip(retrieved_sims, retrieved_indices):
                if ret_idx >= len(gallery_labels):
                    continue  # Skip out-of-range indices
                if sim >= threshold:
                    if labels_match(query_label, gallery_labels[ret_idx]):
                        tp += 1
                    else:
                        fp += 1
            
            total_tp += tp
            total_fp += fp
            total_fn += total_positives[q_idx] - tp
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions[t_idx] = precision
        recalls[t_idx] = recall
        f1_scores[t_idx] = f1
    
    # Find optimal threshold
    best_f1_idx = np.argmax(f1_scores)
    
    # Calculate AUC using trapezoidal rule
    sorted_indices = np.argsort(recalls)
    auc = np.trapz(precisions[sorted_indices], recalls[sorted_indices])
    
    return {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'auc': abs(auc),
        'average_precision': np.mean(precisions),
        'max_accuracy': np.max(f1_scores),
        'optimal_threshold': thresholds[best_f1_idx],
        'optimal_acc_threshold': thresholds[best_f1_idx]
    }


def find_optimal_threshold(binary_metrics, criterion='f1'):
    """
    Find optimal threshold based on criterion.
    
    Args:
        binary_metrics: Dict from calculate_binary_metrics
        criterion: 'f1', 'precision', or 'recall'
        
    Returns:
        Dict with optimal threshold and metrics at that threshold
    """
    if criterion == 'f1':
        scores = binary_metrics['f1_scores']
    elif criterion == 'precision':
        scores = binary_metrics['precisions']
    elif criterion == 'recall':
        scores = binary_metrics['recalls']
    else:
        scores = binary_metrics['f1_scores']
    
    best_idx = np.argmax(scores)
    
    return {
        'threshold': binary_metrics['thresholds'][best_idx],
        'precision': binary_metrics['precisions'][best_idx],
        'recall': binary_metrics['recalls'][best_idx],
        'f1_score': binary_metrics['f1_scores'][best_idx],
        'accuracy': binary_metrics['f1_scores'][best_idx]
    }


def analyze_label_distribution(gallery_labels, query_labels):
    """
    Analyze the distribution of labels in gallery and queries.
    
    Returns:
        Dict with statistics about label distribution
    """
    def get_stats(labels):
        if isinstance(labels[0], (list, np.ndarray)):
            n_labels = [len(l) if hasattr(l, '__len__') else 1 for l in labels]
        else:
            n_labels = [1] * len(labels)
        
        return {
            'total_items': len(labels),
            'avg_labels_per_item': np.mean(n_labels),
            'max_labels_per_item': max(n_labels),
            'min_labels_per_item': min(n_labels)
        }
    
    return {
        'gallery_stats': get_stats(gallery_labels),
        'query_stats': get_stats(query_labels)
    }
