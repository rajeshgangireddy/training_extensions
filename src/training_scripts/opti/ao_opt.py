import os
import sys
from src.otx.engine import Engine

PATH_TO_REPO = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/"

sys.path.append(os.path.join(PATH_TO_REPO, "src"))

data_root = os.path.join(PATH_TO_REPO, "tests/assets/common_semantic_segmentation_dataset")
recipe = os.path.join(PATH_TO_REPO, "src/otx/recipe/semantic_segmentation/litehrnet_18.yaml")

engine = Engine(data_root=data_root,work_dir="otx-workspace")

model = engine.model
print(type(model))




engine.train(max_epochs=2)


import torch
import torchao
import torch.cuda

# def print_gpu_memory():
#     gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
#     print(f"GPU Memory Allocated: {gpu_memory:.2f} MB")
# print_gpu_memory()
#
# from torchao.quantization.quant_api import (
#     quantize_,
#     Int8DynamicActivationInt8WeightConfig,
#     Int4WeightOnlyConfig,
#     Int8WeightOnlyConfig
# )
#
# model_quant = quantize_(model, Int8WeightOnlyConfig())
# print(type(model_quant))
# engine.model = model_quant

model = torchao.autoquant(torch.compile(model, mode='max-autotune'))
print(type(model))

# engine.train(max_epochs=10)

