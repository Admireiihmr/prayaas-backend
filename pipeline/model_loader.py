import urllib.request
from functools import lru_cache

from prayaas.config.configuration import settings
from prayaas.logger import logger
from prayaas.utils.common import ensure_dir


def _download_if_missing() -> None:
    ensure_dir(settings.models_dir)

    for path, url in (
        (settings.model_json_path, settings.model_json_url),
        (settings.model_path, settings.model_url),
    ):
        if path.exists():
            continue
        logger.info("Downloading %s from %s", path.name, url)
        urllib.request.urlretrieve(url, path)


@lru_cache(maxsize=1)
def load_model():
    """Loads the pretrained oral-cancer classifier.

    Uses tf_keras, not keras: the architecture was serialized with Keras 2.15 and
    Keras 3 cannot deserialize it ("Could not locate class 'Functional'").
    """
    _download_if_missing()

    import tf_keras

    logger.info("Loading model architecture from %s", settings.model_json_path.name)
    model = tf_keras.models.model_from_json(settings.model_json_path.read_text(encoding="utf-8"))

    if model is None:
        raise ValueError(
            f"Could not rebuild the model from {settings.model_json_path.name}. "
            f"The file is present but is not a valid Keras architecture."
        )

    logger.info("Loading weights from %s", settings.model_path.name)
    model.load_weights(str(settings.model_path))

    logger.info("Model ready: input=%s output=%s", model.input_shape, model.output_shape)
    return model


def reset_model_cache() -> None:
    load_model.cache_clear()


@lru_cache(maxsize=1)
def load_segmentation_model():
    """Loads the U-Net lesion-segmentation model.

    Unlike load_model(), this is a single .keras file saved with the
    architecture included (ModelCheckpoint's default), so it loads directly
    -- no separate json + weights, no MODEL_URL download fallback. If the
    file isn't there, this raises and the caller treats segmentation as
    unavailable rather than failing the whole prediction.
    """
    import tf_keras

    logger.info("Loading segmentation model from %s", settings.unet_model_path.name)
    return tf_keras.models.load_model(str(settings.unet_model_path), compile=False)


def reset_segmentation_model_cache() -> None:
    load_segmentation_model.cache_clear()


if __name__ == "__main__":
    load_model()
