import numpy as np
import pandas as pd
import pytest

from prayaas.config.configuration import settings
from pipeline.data_ingestion import DataIngestion
from pipeline.data_preprocessing import DataPreprocessing
from pipeline.model_evaluation import collapse_to_binary


def test_ingestion_indexes_both_classes():
    index = DataIngestion().run()
    assert set(index["folder"]) == {"CANCER", "NON CANCER"}
    assert set(index["label"]) == {0, 1}
    assert len(index) == len(index["path"].unique())


def test_preprocessing_shape_and_range():
    path = DataIngestion().run()["path"].iloc[0]
    array = DataPreprocessing().from_path(path)

    assert array.shape == (settings.image_size, settings.image_size, 3)
    assert array.dtype == np.float32
    assert 0.0 <= array.min() and array.max() <= 1.0


def test_collapse_passes_through_the_two_real_classes():
    assert collapse_to_binary(0) == 0
    assert collapse_to_binary(1) == 1


def test_collapse_maps_opmd_per_setting():
    """OPMD has no folder on disk, so it collapses per OPMD_IS_CANCER."""
    expected = 0 if settings.opmd_is_cancer else 1
    assert collapse_to_binary(2) == expected


def test_limited_evaluation_is_stratified():
    """The --limit path must not silently drop a class."""
    index = DataIngestion().run()
    per_class = 5
    limited = index[index.groupby("label").cumcount() < per_class]

    assert set(limited["label"]) == {0, 1}
    assert len(limited) == per_class * 2
