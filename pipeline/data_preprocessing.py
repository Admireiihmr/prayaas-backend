import cv2
import numpy as np
from PIL import Image

from prayaas.config.configuration import settings
from prayaas.logger import logger


def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    """Per-channel CLAHE contrast enhancement.

    Mirrors the enhancement the original Streamlit client applied before upload,
    so training-time and serving-time inputs look the same.
    """
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    clahe = cv2.createCLAHE(clipLimit=settings.clahe_clip_limit, tileGridSize=(8, 8))
    merged = cv2.merge([clahe.apply(channel) for channel in cv2.split(bgr)])
    return cv2.cvtColor(merged, cv2.COLOR_BGR2RGB)


class DataPreprocessing:
    """Turns an image into the (224, 224, 3) float array the model expects."""

    def from_pil(self, image: Image.Image, steps: list | None = None) -> np.ndarray:
        """steps, if given, is appended with (label, PIL.Image) pairs for each
        stage below — used to show the processing breakdown in the UI."""
        image = image.convert("RGB")
        if steps is not None:
            steps.append(("Original", image.copy()))

        image = image.resize((settings.image_size, settings.image_size))
        array = np.array(image, dtype=np.uint8)
        if steps is not None:
            steps.append(("Resized (224×224)", Image.fromarray(array)))

        if settings.apply_clahe:
            array = apply_clahe(array)
            if steps is not None:
                steps.append(("Contrast Enhanced (CLAHE)", Image.fromarray(array)))

        return array.astype(np.float32) / 255.0

    def from_path(self, path: str) -> np.ndarray:
        with Image.open(path) as image:
            return self.from_pil(image)

    def batch_from_paths(self, paths: list[str]) -> np.ndarray:
        return np.stack([self.from_path(p) for p in paths])


if __name__ == "__main__":
    from pipeline.data_ingestion import DataIngestion

    sample = DataIngestion().run()["path"].iloc[0]
    logger.info("Preprocessed %s -> %s", sample, DataPreprocessing().from_path(sample).shape)
