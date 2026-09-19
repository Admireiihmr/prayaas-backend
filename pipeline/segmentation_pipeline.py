import cv2
import numpy as np
from PIL import Image

from pipeline.model_loader import load_segmentation_model
from pipeline.unet_architecture import IMG_SIZE

_RED = (239, 68, 68)  # matches the frontend's "concerning" red

# Below this fraction of the (256x256) mask's area, a connected blob is
# dropped as speckle noise. Measured empirically on the test set: this is
# NOT a meaningful accuracy lever (+0.0002 mean IoU at best, negative at
# higher thresholds -- some "speckles" turned out to be real small lesion
# detections). Kept small and conservative, for visual tidiness only.
_MIN_BLOB_AREA_FRACTION = 0.001


def _drop_small_blobs(mask: np.ndarray, min_area_fraction: float = _MIN_BLOB_AREA_FRACTION) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    min_area = min_area_fraction * mask.shape[0] * mask.shape[1]

    cleaned = np.zeros_like(mask)
    for label in range(1, num_labels):  # label 0 is background
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def predict_mask(image: Image.Image, threshold: float = 0.5) -> np.ndarray:
    """Runs the U-Net and returns the predicted lesion mask at IMG_SIZE x
    IMG_SIZE (uint8, 0 or 255). Callers resize it to whatever resolution they
    need -- e.g. segment_lesion() below upscales to the original image size,
    while Grad-CAM masking (prediction_pipeline.py) resizes to the
    classifier's 224x224 input instead.
    """
    resized = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0

    pred = load_segmentation_model().predict(array[None, ...], verbose=0)[0, ..., 0]
    mask = (pred > threshold).astype(np.uint8) * 255
    return _drop_small_blobs(mask)


def segment_lesion(image: Image.Image, mask: np.ndarray | None = None) -> Image.Image:
    """Returns the original image with the predicted lesion region overlaid
    in red, upscaled back to its size.

    Pass a mask already computed via predict_mask() to avoid a second
    forward pass through the model; otherwise one is computed here.

    Mirrors segment_image() in src/prayaas/research/segmentation.ipynb.
    """
    original = image.convert("RGB")
    if mask is None:
        mask = predict_mask(image)
    mask_image = Image.fromarray(mask).resize(original.size, Image.NEAREST)

    overlay_layer = Image.new("RGBA", original.size, (*_RED, 0))
    overlay_layer.putalpha(mask_image.point(lambda p: 120 if p > 0 else 0))
    return Image.alpha_composite(original.convert("RGBA"), overlay_layer).convert("RGB")
