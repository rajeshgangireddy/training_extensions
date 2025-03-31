# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX classification perfomance benchmark tests."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from .benchmark import Benchmark
from .conftest import PerfTestBase

log = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def fxt_deterministic(request: pytest.FixtureRequest) -> bool:
    """Override the deterministic setting for classification tasks."""
    deterministic = request.config.getoption("--deterministic")
    deterministic = True if deterministic is None else deterministic == "true"
    log.info(f"{deterministic=}")
    return deterministic


class TestPerfSingleLabelClassification(PerfTestBase):
    """Benchmark single-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/multi_class_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/multi_class_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/multi_class_cls", name="deit_tiny", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="dino_v2", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_b3", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_efficientnet_v2_l", category="other"),
        Benchmark.Model(task="classification/multi_class_cls", name="tv_mobilenet_v3_small", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name="multiclass_Pneumonia_tiny",
            path=Path("Processed/ChestXRay-Pneumonia_12_6_200_UNIFORM_UNIFORM_TESTSET"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_CUB-Woodpecker_tiny",
            path=Path("Processed/CUB-WOODPECKER_24_12_200_UNIFORM_UNIFORM_TESTSET"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_DTD_tiny",
            path=Path("Processed/DTD_24_6_200_UNIFORM_UNIFORM_TESTSET"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_IntelImg_small",
            path=Path("Processed/IntelImgCls_50_10_200_UNIFORM_UNIFORM_TESTSET"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_flowers_small",
            path=Path("Processed/Flowers_60_12_200_STRAT_UNIFORM-TESTSET"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_EuroSat_small",
            path=Path("Processed/EuroSat_80_40_200_STRAT_UNIFORM-TESTSET"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_Dogs_medium",
            path=Path("Processed/StanfordDogs-Terrier_240_60_400_STRAT_UNIFORM-TESTSET"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_CheXpert_medium",
            path=Path("Processed/CheXpert_240_60_400_STRAT_UNIFORM-TESTSET-TESTSET"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_RESISC45_medium",
            path=Path("Processed/RESISC45_500_100_400_STRAT_UNIFORM-TESTSET"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_CUB100_large",
            path=Path("Processed/CUB_3764_900_1200_STRAT_UNIFORM-TESTSET_NC-100"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multiclass_CUB_large",
            path=Path("Processed/CUB_7510_1878_2400_STRAT_UNIFORM-TESTSET"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
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


class TestPerfMultiLabelClassification(PerfTestBase):
    """Benchmark multi-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/multi_label_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/multi_label_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/multi_label_cls", name="deit_tiny", category="other"),
        # Benchmark.Model(task="classification/multi_label_cls", name="dino_v2", category="other"), DINO NOT SUPPORTED
        Benchmark.Model(task="classification/multi_label_cls", name="tv_efficientnet_b3", category="other"),
        Benchmark.Model(task="classification/multi_label_cls", name="tv_efficientnet_v2_l", category="other"),
        Benchmark.Model(task="classification/multi_label_cls", name="tv_mobilenet_v3_small", category="other"),
    ]


    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name="multilabel_BCCD_tiny",
            path=Path("Processed/BCCD_mlabel_tiny_24_6_100"),
            group="tiny",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_coco_small",
            path=Path("Processed/coco_mlabel_small_80_20_100"),
            group="small",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_edsa_medium",
            path=Path("Processed/EDSA_Vehicle_medium_600_150_200"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="multilabel_aid_large",
            path=Path("Processed/AID_mlabel_large_1920_480_600"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]


    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
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


class TestPerfHierarchicalLabelClassification(PerfTestBase):
    """Benchmark hierarchical-label classification."""

    MODEL_TEST_CASES = [  # noqa: RUF012
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_b0", category="speed"),
        Benchmark.Model(task="classification/h_label_cls", name="efficientnet_v2", category="balance"),
        Benchmark.Model(task="classification/h_label_cls", name="mobilenet_v3_large", category="accuracy"),
        Benchmark.Model(task="classification/h_label_cls", name="deit_tiny", category="other"),
    ]

    DATASET_TEST_CASES = [
        Benchmark.Dataset(
            name=f"hlabel_CUB_small_{idx}",
            path=Path("hlabel_classification/hlabel_CUB_small") / f"{idx}",
            group="small",
            num_repeat=5,
            extra_overrides={},
        )
        for idx in (1, 2, 3)
    ] + [
        Benchmark.Dataset(
            name="hlabel_CUB_medium",
            path=Path("hlabel_classification/hlabel_CUB_medium"),
            group="medium",
            num_repeat=5,
            extra_overrides={},
        ),
        Benchmark.Dataset(
            name="cifar100_label_group_datum_format_large",
            path=Path("hlabel_classification/cifar100_label_group_datum_format_large"),
            group="large",
            num_repeat=5,
            extra_overrides={},
        ),
    ]

    BENCHMARK_CRITERIA = [  # noqa: RUF012
        Benchmark.Criterion(name="train/epoch", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="train/e2e_time", summary="max", compare="<", margin=0.1),
        Benchmark.Criterion(name="val/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="test/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="export/accuracy", summary="max", compare=">", margin=0.1),
        Benchmark.Criterion(name="optimize/accuracy", summary="max", compare=">", margin=0.1),
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
