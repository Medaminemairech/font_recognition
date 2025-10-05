import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm


def generate_metadata(base_dir: str, output_path: str) -> None:
    """Generates a CSV file containing metadata for images in the specified directory structure.
    Args:
        base_dir (str): The base directory containing class subdirectories with images.
        output_path (str): The directory where the output CSV file will be saved.
    Returns:
        pd.DataFrame: A DataFrame containing the metadata of the images.
    """
    data = []
    for class_name in tqdm(os.listdir(base_dir)):
        class_path = os.path.join(base_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            if not (file_name.lower().endswith((".png", ".jpg", ".jpeg"))):
                continue

            try:
                with Image.open(file_path) as img:
                    width, height = img.size
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                continue

            data.append(
                {
                    "id": len(data),
                    "path": file_path,
                    "class": class_name,
                    "width": width,
                    "height": height,
                }
            )
    df = pd.DataFrame(data)

    artifacts_dir = os.path.join(output_path, f"metadata/")
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{artifacts_dir}/font_dataset.csv", index=False)
    return df
