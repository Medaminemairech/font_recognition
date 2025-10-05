from collections import defaultdict

import numpy as np
from PIL import Image


def safe_loader(path):
    """
    Safely loads an image from the given path.
    If the image is corrupted, returns None and prints a warning.
    """
    try:
        with Image.open(path) as img:
            return img.convert("RGB")
    except Exception as e:
        print(f"Skipping corrupted image: {path} ({e})")
        return None


def balanced_subset_indices(
    dataset, n_per_class=None, fraction_per_class=None, seed=42
):
    """
    Returns indices for a balanced subset of an ImageFolder dataset.

    Args:
        dataset: torchvision.datasets.ImageFolder
        n_per_class: number of samples per class (int)
        fraction_per_class: fraction of samples per class (float, 0–1)
        seed: random seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    class_to_indices = defaultdict(list)

    # Group indices by class
    for idx, (_, label) in enumerate(dataset.samples):
        class_to_indices[label].append(idx)

    selected_indices = []
    for label, indices in class_to_indices.items():
        if fraction_per_class is not None:
            n_samples = int(len(indices) * fraction_per_class)
        else:
            n_samples = min(n_per_class, len(indices))
        chosen = rng.choice(indices, n_samples, replace=False)
        selected_indices.extend(chosen)

    return selected_indices
