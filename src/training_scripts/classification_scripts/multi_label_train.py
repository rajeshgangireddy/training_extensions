from src.otx.engine import Engine
import os


recipe_full_path = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/src/otx/recipe/classification/multi_label_cls/efficientnet_b0.yaml"
# data_path_full = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/tests/assets/multilabel_classification"
data_path_full = "/home/rgangire/workspace/datasets/Classification/Processed/coco_mlabel_small_80_20_100"

engine = Engine.from_config(
          config_path=recipe_full_path,
          data_root=data_path_full,
          work_dir="otx-workspace",
        )

engine.train()
engine.test()