import base64
import io

import numpy as np
from PIL import Image

from prayaas.config.configuration import CLASS_LABELS, settings
from prayaas.logger import logger
from pipeline.data_preprocessing import DataPreprocessing
from pipeline.gradcam import compute_heatmap, overlay_heatmap
from pipeline.model_loader import load_model
from pipeline.segmentation_pipeline import predict_mask, segment_lesion


class PredictionPipeline:
    """Scores images against the pretrained classifier."""

    def __init__(self) -> None:
        self.preprocessor = DataPreprocessing()

    def predict_array(self, batch: np.ndarray) -> np.ndarray:
        """batch: (n, 224, 224, 3) float in [0, 1] -> (n, 3) probabilities."""
        return load_model().predict(batch, verbose=0)

    def predict_image(self, image: Image.Image) -> dict:
        steps = []
        array = self.preprocessor.from_pil(image, steps=steps)
        batch = np.expand_dims(array, axis=0)
        probabilities = self.predict_array(batch)[0]
        result = self._format(probabilities)

        # Highlighting a "lesion" on an image the classifier itself calls
        # normal is misleading, not informative -- both explainability
        # steps only make sense when something was actually found.
        if result["label"] != "Normal":
            lesion_mask = None
            try:
                lesion_mask = self._append_segmentation_steps(steps)
            except Exception:
                # Segmentation is a separate, optionally-trained model --
                # missing or failing must not cost the classification result.
                logger.exception("U-Net segmentation failed")

            try:
                self._append_gradcam_steps(steps, batch, result["predicted_class"], lesion_mask)
            except Exception:
                # The classification result must still ship even if the
                # explainability overlay fails to render.
                logger.exception("Grad-CAM heatmap generation failed")

        result["processing_steps"] = [
            {"label": label, "image": self._encode_png(step_image)} for label, step_image in steps
        ]
        return result

    @staticmethod
    def _append_segmentation_steps(steps: list) -> np.ndarray:
        """Appends the U-Net's predicted lesion region, overlaid on the
        full-resolution original -- shown before classification, matching the
        segment-then-classify pipeline PRAYAAS's paper describes.

        Returns the raw mask (256x256) so Grad-CAM can be confined to it."""
        original_image = steps[0][1]  # ("Original", full-res PIL image)
        mask = predict_mask(original_image)
        steps.append(("Lesion Segmentation (U-Net)", segment_lesion(original_image, mask=mask)))
        return mask

    @staticmethod
    def _append_gradcam_steps(
        steps: list, batch: np.ndarray, predicted_class: int, lesion_mask: np.ndarray | None = None
    ) -> None:
        """Appends the Grad-CAM heatmap blended onto the actual classifier
        input, showing which regions drove the prediction. Derives its base
        image from `batch` directly (not from `steps`) -- the CLAHE-processed
        224x224 array the model actually saw is the correct overlay base
        regardless of what's in the display-only `steps` list.

        If `lesion_mask` (from the U-Net, 256x256) is given, the heatmap is
        confined to that region -- rather than the two explainability
        signals coloring the image independently and sometimes disagreeing,
        Grad-CAM only highlights within what the U-Net calls the lesion."""
        base_rgb = (batch[0] * 255).astype(np.uint8)
        heatmap = compute_heatmap(load_model(), batch, predicted_class)

        mask_resized = None
        if lesion_mask is not None:
            size = (base_rgb.shape[1], base_rgb.shape[0])  # (width, height)
            mask_resized = np.array(Image.fromarray(lesion_mask).resize(size, Image.NEAREST))

        overlay = overlay_heatmap(heatmap, base_rgb, mask=mask_resized)
        steps.append(("Heatmap Overlay (Suspicious Regions)", Image.fromarray(overlay)))

    def predict_bytes(self, raw: bytes) -> dict:
        with Image.open(io.BytesIO(raw)) as image:
            return self.predict_image(image)

    def predict_base64(self, encoded: str) -> dict:
        """Accepts a bare base64 string or a data: URL, matching the old client."""
        if "," in encoded and encoded.strip().startswith("data:"):
            encoded = encoded.split(",", 1)[1]
        return self.predict_bytes(base64.b64decode(encoded))

    def predict_paths(self, paths: list[str]) -> list[dict]:
        results = []
        for start in range(0, len(paths), settings.batch_size):
            chunk = paths[start : start + settings.batch_size]
            batch = self.preprocessor.batch_from_paths(chunk)
            results.extend(self._format(p) for p in self.predict_array(batch))
            logger.info("Scored %s/%s images", min(start + len(chunk), len(paths)), len(paths))
        return results

    @staticmethod
    def _encode_png(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

    @staticmethod
    def _format(probabilities: np.ndarray) -> dict:
        index = int(np.argmax(probabilities))
        return {
            "predicted_class": index,
            "label": CLASS_LABELS[index],
            "confidence": round(float(probabilities[index]), 4),
            # Percentages, matching the original API's response contract.
            "probabilities": {
                label: round(float(p) * 100, 2) for label, p in zip(CLASS_LABELS, probabilities)
            },
        }
