import pandas as pd

from prayaas.config.configuration import FOLDER_TO_CLASS, settings
from prayaas.logger import logger
from prayaas.utils.common import ensure_dir

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


class DataIngestion:
    """Indexes the class folders under artifacts/raw_data into a path/label table."""

    def run(self) -> pd.DataFrame:
        ensure_dir(settings.raw_data_dir)

        rows = []
        for folder, label in FOLDER_TO_CLASS.items():
            directory = settings.raw_data_dir / folder
            if not directory.is_dir():
                logger.warning("Missing class folder: %s", directory)
                continue

            paths = [p for p in sorted(directory.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
            rows.extend({"path": str(p), "label": label, "folder": folder} for p in paths)
            logger.info("%s: %s images -> class %s", folder, len(paths), label)

        if not rows:
            raise FileNotFoundError(
                f"No images found under {settings.raw_data_dir}. "
                f"Expected class folders: {list(FOLDER_TO_CLASS)}"
            )

        df = pd.DataFrame(rows)
        df.to_csv(settings.image_index_path, index=False)
        logger.info("Indexed %s images -> %s", len(df), settings.image_index_path.name)
        return df


if __name__ == "__main__":
    DataIngestion().run()
