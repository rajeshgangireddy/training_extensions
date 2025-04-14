from src.otx.engine import Engine
import time

recipe_full_path = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/src/otx/recipe/classification/h_label_cls/efficientnet_b0.yaml"
data_path_full = "/home/rgangire/workspace/datasets/Classification/Processed/CardsDataset-H-Label-Tiny-2L-6N_36_20_100"

engine = Engine.from_config(
          config_path=recipe_full_path,
          data_root=data_path_full,
          work_dir="otx-workspace",
        )
tic = time.time()
engine.train()
print(f"Training time: {time.time() - tic}")

engine.test()