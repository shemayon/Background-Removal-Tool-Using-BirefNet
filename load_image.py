import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np

def load_img(source, output_type="pil"):
    """
    Load an image from a local file path or a URL.

    Parameters:
    - source (str): A file path or a URL.
    - output_type (str): The output format: "pil" (PIL Image) or "numpy" (NumPy array).

    Returns:
    - PIL.Image.Image or numpy.ndarray depending on output_type.
    """

    # Determine if `source` is a local file path or a URL
    if os.path.exists(source):
        # Local file
        img = Image.open(source)
    else:
        # Assume source is a URL
        response = requests.get(source)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

    if output_type == "pil":
        return img
    elif output_type == "numpy":
        return np.array(img)
    else:
        raise ValueError(f"Unknown output_type: {output_type}")
