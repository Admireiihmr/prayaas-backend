import cv2
import numpy as np
import tensorflow as tf


def compute_heatmap(model, batch: np.ndarray, class_index: int, layer_name: str = "relu") -> np.ndarray:
    """Grad-CAM: (H, W) heatmap in [0, 1] showing which regions of the last conv
    block's feature map pushed the prediction toward class_index."""
    import tf_keras

    grad_model = tf_keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(batch)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def colorize(heatmap: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resizes a [0, 1] heatmap to (height, width) and applies a jet colormap."""
    resized = cv2.resize(heatmap, (size[1], size[0]))
    heat_uint8 = np.uint8(255 * resized)
    colored_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)


def overlay_heatmap(
    heatmap: np.ndarray, base_rgb: np.ndarray, alpha: float = 0.45, mask: np.ndarray | None = None
) -> np.ndarray:
    """Blends the colorized heatmap over the base image, highlighting the
    regions the model weighted most heavily for its prediction.

    `mask`, if given (same H, W as base_rgb; nonzero = keep), confines the
    blend to those pixels -- elsewhere the base image shows through
    unchanged. Lets a caller align Grad-CAM's visualization to e.g. a lesion
    segmentation mask instead of coloring the whole image.
    """
    colored_rgb = colorize(heatmap, base_rgb.shape[:2])
    alpha_map = np.full(base_rgb.shape[:2], alpha, dtype=np.float32)
    if mask is not None:
        alpha_map *= (mask > 0).astype(np.float32)
    alpha_map = alpha_map[..., None]
    blended = colored_rgb.astype(np.float32) * alpha_map + base_rgb.astype(np.float32) * (1 - alpha_map)
    return np.clip(blended, 0, 255).astype(np.uint8)
