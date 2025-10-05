import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .transforms import CropTextBlockTransform


class TestImageDataset(Dataset):
    """Custom Dataset for loading test images from a directory.
    Assumes images are in formats supported by PIL (e.g., .png, .jpg).

    Args:
        image_dir (str): Directory with all the test images.
        transform (callable, optional): Optional transform to be applied on an image.

        Returns:
        image (Tensor): Transformed image tensor.
        filename (str): Original filename of the image.
    """

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(img_path)  # filename for reference


class TestDataLoader:
    """
    Wrapper class to create a DataLoader for the test dataset,
    cropping the text block first.
    Args:
        test_dir (str): Directory containing test images.
        batch_size (int): Number of images per batch.
        img_size (int): Size to which each image will be resized (img_size x img_size).
    Returns:
        DataLoader: PyTorch DataLoader for the test dataset.
    """

    def __init__(self, test_dir: str, batch_size: int = 32, img_size: int = 224):
        self.test_dir = test_dir
        self.batch_size = batch_size

        # Compose transforms: first crop text block, then resize and normalize
        self.transform = transforms.Compose(
            [
                CropTextBlockTransform(lang_list=["en"]),  # <--- first transform
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.dataset = TestImageDataset(test_dir, transform=self.transform)
        self.loader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)
