import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from training.config import DataConfig

from .utils import balanced_subset_indices, safe_loader


class ImageDataModule:
    def __init__(self, config: DataConfig):
        """
        Initializes the data module with configuration parameters.
        Args:
            config (DataConfig): Configuration object containing data parameters.
        returns: None
        """
        self.config = config
        self._prepare_transforms()

    def _prepare_transforms(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.config.img_size, self.config.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def _load_dataset(self):
        self.full_dataset = datasets.ImageFolder(
            self.config.data_dir, transform=self.transform, loader=safe_loader
        )

    def _split_dataset(self):

        # If subset_fraction is set, sample a balanced subset
        if self.config.subset_fraction is not None:
            subset_indices = balanced_subset_indices(
                self.full_dataset,
                fraction_per_class=self.config.subset_fraction,
                seed=42,
            )
            self.dataset = torch.utils.data.Subset(self.full_dataset, subset_indices)
        else:
            self.dataset = self.full_dataset

        dataset_size = len(self.dataset)
        train_size = int(self.config.train_split * dataset_size)
        val_size = dataset_size - train_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, val_size]
        )

    def _create_loaders(self):
        train_loader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        return train_loader, val_loader

    def setup_training(self):
        """
        Prepares the datasets and dataloaders.
        """
        self._load_dataset()
        self._split_dataset()
        return self._create_loaders()

    @property
    def summary(self):
        print(f"Train: {len(self.train_dataset)}, Validation: {len(self.val_dataset)}")

    @property
    def num_classes(self):
        return len(self.full_dataset.classes)

    @property
    def classes_dict(self):
        return self.full_dataset.class_to_idx
