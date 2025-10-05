import easyocr
import numpy as np
from PIL import Image


class CropTextBlockTransform:
    """
    Transform to crop the entire text block from an image using EasyOCR.
    If no text is detected, returns the original image.
    Uses PIL for image handling.
    args:
        lang_list: list of languages for EasyOCR (default is ["en"])
    returns: cropped PIL Image
    """

    def __init__(self, lang_list=["en"]):
        self.reader = easyocr.Reader(lang_list, gpu=False)  # set gpu=True if available

    def __call__(self, img):
        """
        img: PIL Image
        returns: cropped PIL Image containing the full text block
        """
        if not isinstance(img, Image.Image):
            raise TypeError("Input must be a PIL Image")

        # Convert PIL to NumPy array for EasyOCR
        img_np = np.array(img)

        # Detect text
        results = self.reader.readtext(img_np)

        if len(results) == 0:
            # No text detected, return original image
            return img

        # Combine all bounding boxes
        x_min = min([int(bbox[0][0]) for bbox, _, _ in results])
        y_min = min([int(bbox[0][1]) for bbox, _, _ in results])
        x_max = max([int(bbox[2][0]) for bbox, _, _ in results])
        y_max = max([int(bbox[2][1]) for bbox, _, _ in results])

        # Crop the block using PIL
        cropped_block = img.crop((x_min, y_min, x_max, y_max))
        return cropped_block
