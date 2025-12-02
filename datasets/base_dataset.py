"""
Base dataset class for retrieval datasets.
"""

import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class BaseDataset(Dataset):
    """
    Base dataset class for instance retrieval.
    
    Expected annotation format (PyTorch .pt file):
    {
        "/path/to/image.jpg": {
            "bbox": [x1, y1, x2, y2],  # Optional bounding box
            "ins": 0,                   # Instance label
            "is_query": True/False,     # Query or gallery
            "obj_name": "name"          # Optional object name
        },
        ...
    }
    """
    
    def __init__(
        self,
        root=None,
        ann_file=None,
        transform=None,
        format='XYXY',
        args=None,
        only_queries=False,
        open_pil=True,
        instance_from_name=True,
        ds_inds_file=None
    ):
        """
        Initialize the dataset.
        
        Args:
            root: Root directory for images (optional, paths can be absolute)
            ann_file: Path to annotation file (.pt format)
            transform: Optional transform to apply
            format: Bounding box format ('XYXY', 'XYWH', 'POLY')
            args: Additional arguments
            only_queries: Only load query images
            open_pil: Whether to open images as PIL
            instance_from_name: Whether to derive instance from filename
            ds_inds_file: Path to file with dataset indices to use (subset)
        """
        self.root = Path(root) if root else None
        self.ann_file = Path(ann_file) if ann_file else None
        self.ds_inds_file = ds_inds_file
        self.transform = transform
        self.format = format
        self.args = args
        self.only_queries = only_queries
        self.open_pil = open_pil
        self.instance_from_name = instance_from_name
        
        # Data structures
        self.anns = {}
        self.imgs_paths_ls = []
        self.imgs_paths_ls_query = []
        self.imgs_paths_ls_gallery = []
        self.labels = []
        self.cls_names = []
        self.bboxes = []
        
        # Load annotations
        if self.ann_file is not None:
            self._load_annotations()
    
    def _load_annotations(self):
        """Load annotations from file."""
        print(f"Loading annotations from {self.ann_file}...")
        self.anns = torch.load(self.ann_file)
        
        # Filter by ds_inds_file if provided
        if self.ds_inds_file is not None:
            print(f"Filtering by indices from {self.ds_inds_file}...")
            ds_inds = torch.load(self.ds_inds_file)
            im_paths = list(self.anns.keys())
            self.anns = {im_paths[_ind]: self.anns[im_paths[_ind]] for _ind in ds_inds}
        
        print("Indexing dataset...")
        for img_path in tqdm(self.anns):
            ann = self.anns[img_path]
            
            is_query = ann.get('is_query', False)
            
            # Skip gallery if only_queries is set
            if self.only_queries and not is_query:
                continue
            
            bbox = ann.get('bbox', None)
            label = ann.get('ins', ann.get('label', 0))
            
            self._add_item(
                is_query=is_query,
                bbox=bbox,
                im_path=str(img_path),
                label=label
            )
    
    def _add_item(self, is_query, bbox, im_path, label):
        """Add an item to the dataset."""
        self.imgs_paths_ls.append(im_path)
        
        if is_query:
            self.imgs_paths_ls_query.append(im_path)
        else:
            self.imgs_paths_ls_gallery.append(im_path)
        
        self.labels.append(label)
        self.bboxes.append(bbox)
        
        # Extract class name from path or annotation
        if self.instance_from_name:
            cls_name = Path(im_path).stem.split('_')[0]
        else:
            cls_name = str(label)
        self.cls_names.append(cls_name)
    
    def __len__(self):
        return len(self.imgs_paths_ls)
    
    def __getitem__(self, index):
        """
        Get an item from the dataset.
        
        Returns:
            img_pil: PIL Image
            img_path: Path to the image
            (index, bbox, label, scale): Metadata tuple
        """
        img_path = self.imgs_paths_ls[index]
        
        # Load image
        if self.open_pil:
            img_pil = Image.open(img_path).convert('RGB')
        else:
            img_pil = img_path
        
        # Apply transform if provided
        if self.transform is not None:
            img_pil = self.transform(img_pil)
        
        bbox = self.bboxes[index]
        label = self.labels[index]
        scale = 1.0
        
        return img_pil, img_path, (index, bbox, label, scale)
    
    def is_query(self, index):
        """Check if an item is a query."""
        img_path = self.imgs_paths_ls[index]
        return self.anns[img_path].get('is_query', False)
    
    def get_cls_name(self, index):
        """Get class name for an item."""
        return self.cls_names[index]
    
    @property
    def num_queries(self):
        """Number of query images."""
        return len(self.imgs_paths_ls_query)
    
    @property
    def num_gallery(self):
        """Number of gallery images."""
        return len(self.imgs_paths_ls_gallery)


def get_dataset_class(args):
    """
    Get dataset class and kwargs based on configuration.
    
    Args:
        args: Configuration with 'dataset' and 'anns_file' fields
        
    Returns:
        dataset_class: Dataset class to use
        dataset_kwargs: Keyword arguments for dataset initialization
    """
    dataset_name = getattr(args, 'dataset', 'custom')
    anns_file = getattr(args, 'anns_file', None)
    ds_inds_file = getattr(args, 'ds_inds_file', None)
    
    # Handle dataset-specific defaults
    if dataset_name == 'INSTRE' and anns_file is None:
        anns_file = '/datasets/agamotto/Agamotto-SO/INSTRE/anns.pt'
    elif dataset_name == 'INSTRE-M' and anns_file is None:
        anns_file = '/datasets/agamotto/Agamotto-SO/INSTRE/anns_m.pt'
    elif dataset_name == 'INSTRE-ALL' and anns_file is None:
        anns_file = '/datasets/agamotto/Agamotto-SO/INSTRE/anns_all.pt'
    
    # Default kwargs
    ds_kwargs = {
        'root': None,
        'ann_file': anns_file,
        'transform': None,
        'format': 'XYXY',
        'args': args,
        'ds_inds_file': ds_inds_file,
    }
    
    return BaseDataset, ds_kwargs
