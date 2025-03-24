# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX semantic segmentation perfomance benchmark tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase


class TestPerfSemanticSegmentation(PerfTestBase):
    """Benchmark semantic segmentation."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_18", category="balance"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_s", category="speed"),
        Benchmark.Model(task="semantic_segmentation", name="litehrnet_x", category="accuracy"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_b", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_s", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="segnext_t", category="other"),
        Benchmark.Model(task="semantic_segmentation", name="dino_v2", category="other"),
    ]

    DATASET_TEST_CASES =  [
        Benchmark.Dataset(
            name="green_orange_6_6",
            path=Path("semantic_seg/green_orange_6_6"),
            group="tiny_green_orange_6",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="aerial_200_60",
            path=Path("semantic_seg/aerial_200_60"),
            group="medium_aerial_200",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="aerial_subset_50_20_30",
            path=Path("semantic_seg/aerial_subset_50_20_30"),
            group="small_aerial_50",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="cell_labels_6_6",
            path=Path("semantic_seg/cell_labels_6_6"),
            group="tiny_cells_6",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="flood_segmentation_48_16_16",
            path=Path("semantic_seg/flood_segmentation_48_16_16"),
            group="small_flood_48",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="human_railway_animal_6_6",
            path=Path("semantic_seg/human_railway_animal_6_6"),
            group="tiny_hra",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="idd20k_LITE_1122_204_281",
            path=Path("semantic_seg/idd20k_LITE_1122_204_281"),
            group="large_idd1122",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="kitti_150_50",
            path=Path("semantic_seg/kitti_150_50"),
            group="medium_kitti_150",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="kvasir_large_880_60",
            path=Path("semantic_seg/kvasir_large_880_60"),
            group="large_kvasir_880",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="satellite_buildings_20_8_12",
            path=Path("semantic_seg/satellite_buildings_20_8_12"),
            group="small_satellite_20",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="voc_otx_cut_662_300",
            path=Path("semantic_seg/voc_otx_cut_662_300"),
            group="medium_voc_662",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/Dice", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="train/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="export/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="optimize/iter_time", summary="mean", compare="<", margin=0.1),
        Benchmark.Criterion(name="test(train)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(export)/e2e_time", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test(optimize)/e2e_time", summary="max", compare=">", margin=0.1),
    ]

    @pytest.mark.parametrize(
        "fxt_model",
        MODEL_TEST_CASES,
        ids=lambda model: model.name,
        indirect=True,
    )
    @pytest.mark.parametrize(
        "fxt_dataset",
        DATASET_TEST_CASES,
        ids=lambda dataset: dataset.name,
        indirect=True,
    )
    def test_perf(
        self,
        fxt_model: Benchmark.Model,
        fxt_dataset: Benchmark.Dataset,
        fxt_benchmark: Benchmark,
        fxt_accelerator: str,
    ):
        self._test_perf(
            model=fxt_model,
            dataset=fxt_dataset,
            benchmark=fxt_benchmark,
            criteria=self.BENCHMARK_CRITERIA,
        )
