import cv2
import numpy as np

# ADE20K palette colors
ADE20K_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (120, 120, 120),  # background -> gray
    1: (140, 140, 215),  # crosswalk  -> blue-gray
    2: (180, 120, 120),  # tennis     -> reddish
    3: (61, 230, 250),  # pool       -> cyan
    4: (4, 200, 3),  # eucalyptus -> green
}


def mask_to_ade20k_rgb(mask: np.ndarray) -> np.ndarray:
    """
    Convert grayscale mask (0-4) to ADE20K RGB image for ControlNet-Seg.
    """
    if mask.max() > 4:
        raise ValueError(f"Mask contains invalid class ID {mask.max()}. Did you forget to call export_mask()?")

    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in ADE20K_PALETTE.items():
        rgb[mask == class_id] = color

    return rgb


def mask_to_canny_edges(mask: np.ndarray) -> np.ndarray:
    """
    Generate structural edge map from class boundaries.
    """
    # Spread class IDs (0, 1, 2, 3, 4) into a visible range for Canny
    scaled = (mask * 50).astype(np.uint8)
    edges = cv2.Canny(scaled, threshold1=50, threshold2=150)
    return edges