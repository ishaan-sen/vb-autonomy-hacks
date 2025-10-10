import os, pickle, random
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.model_selection import train_test_split

class AiderPickleDataset(Dataset):
    """
    Loads multiple pickle files (one per class).
    Each pickle payload structure:
      {
        "class_name": str,
        "images": np.ndarray (N, H, W, 3) uint8,
        "paths": List[str],
        "size": (W,H),
        "format": "HWC_uint8_RGB_256x256"
      }
    """
    def __init__(self, samples, transform=None):
        self.samples = samples  # list of (np.ndarray image HWC, int label)
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        # to PIL for torchvision transforms
        from PIL import Image
        pil = Image.fromarray(img, mode="RGB")
        if self.transform: tensor = self.transform(pil)
        else:
            from torchvision import transforms as T
            tensor = T.ToTensor()(pil)
        return tensor, label

def _load_one_pickle(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    imgs = payload["images"]  # (N,H,W,3) uint8
    assert imgs.ndim == 4 and imgs.shape[-1] == 3, f"Bad shape in {pkl_path}"
    return imgs

def load_all_pickles(data_dir: str, class_names: List[str]) -> Dict[str, np.ndarray]:
    data = {}
    for cname in class_names:
        pkl = Path(data_dir) / f"{cname}.pkl"
        arr = _load_one_pickle(pkl)
        data[cname] = arr
    return data

def make_splits(arr_by_class: Dict[str, np.ndarray], class_names: List[str],
                val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Returns: (train_samples, val_samples, test_samples)
    where each is a list of (img, label)
    """
    rng = np.random.RandomState(seed)
    train, val, test = [], [], []
    for label, cname in enumerate(class_names):
        arr = arr_by_class[cname]
        idx = np.arange(len(arr))
        # first split test
        idx_trainval, idx_test = train_test_split(idx, test_size=test_ratio, random_state=seed, stratify=None)
        # then split val from trainval
        vr = val_ratio / (1.0 - test_ratio)
        idx_train, idx_val = train_test_split(idx_trainval, test_size=vr, random_state=seed, stratify=None)

        train += [(arr[i], label) for i in idx_train]
        val   += [(arr[i], label) for i in idx_val]
        test  += [(arr[i], label) for i in idx_test]
    # shuffle
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return train, val, test

def build_transforms(image_size=256, aug=True, label_smoothing=0.05):
    # torchvision >= 0.15 recommended
    tf_train = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.GaussianBlur(kernel_size=3, sigma=(0.1,2.0)),
        T.ToTensor()
    ])
    tf_eval = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor()
    ])
    return tf_train, tf_eval
