"""
Dataset classes for the retrieval system.
"""

__all__ = ['BaseDataset', 'get_dataset_class']

# Lazy imports
def __getattr__(name):
    if name in ('BaseDataset', 'get_dataset_class'):
        from datasets.base_dataset import BaseDataset, get_dataset_class
        if name == 'BaseDataset':
            return BaseDataset
        return get_dataset_class
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
