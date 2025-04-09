from src.otx.engine import Engine
import time

recipe_full_path = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/src/otx/recipe/classification/h_label_cls/efficientnet_b0.yaml"
data_path_full = "/home/rgangire/workspace/datasets/Classification/Processed/Stanford-cars-hlabel_500_50_100"
# data_path_full = "../../tests/assets/hlabel_classification"

engine = Engine.from_config(
          config_path=recipe_full_path,
          data_root=data_path_full,
          work_dir="otx-workspace",
        )
tic = time.time()
engine.train(max_epochs=10)
print(f"Training time: {time.time() - tic}")

engine.test()