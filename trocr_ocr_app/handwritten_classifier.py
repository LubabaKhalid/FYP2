from PIL import Image
import numpy as np

def is_handwritten(image: Image.Image) -> bool:
    """
    Dummy heuristic: checks if the image is noisy (handwritten tends to have uneven strokes).
    For production: Replace with ML classifier or use OCR confidence score.
    """
    gray = image.convert("L").resize((128, 128))
    arr = np.array(gray)
    std_dev = np.std(arr)

    # If standard deviation is high, assume it's handwritten
    return std_dev > 55
