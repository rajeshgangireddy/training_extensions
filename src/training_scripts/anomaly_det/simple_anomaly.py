from otx.engine import Engine

data_root = "/home/rgangire/workspace/datasets/Anomalib/anomaly/mvtec/mvtec/mvtec_bottle_medium"
recipe = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/src/otx/recipe/anomaly_detection/padim.yaml"

engine = Engine.from_config(
          config_path=recipe,
          data_root=data_root,
          work_dir="otx-workspace",
        )

engine.train()