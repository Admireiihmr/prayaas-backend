import numpy as np
from PIL import Image

from pipeline.model_loader import load_segmentation_model
from pipeline.unet_architecture import IMG_SIZE

_RED = (239, 68, 68)  # matches the frontend's "concerning" red


def segment_lesion(image: Image.Image, threshold: float = 0.5) -> Image.Image:
    """Runs the U-Net on a full-resolution image and returns the original with
    the predicted lesion region overlaid in red, upscaled back to its size.

    Mirrors segment_image() in src/prayaas/research/segmentation.ipynb.
    """
    original = image.convert("RGB")
    resized = original.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0

    pred = load_segmentation_model().predict(array[None, ...], verbose=0)[0, ..., 0]
    mask = (pred > threshold).astype(np.uint8) * 255
    mask_image = Image.fromarray(mask).resize(original.size, Image.NEAREST)

    overlay_layer = Image.new("RGBA", original.size, (*_RED, 0))
    overlay_layer.putalpha(mask_image.point(lambda p: 120 if p > 0 else 0))
    return Image.alpha_composite(original.convert("RGBA"), overlay_layer).convert("RGB")
