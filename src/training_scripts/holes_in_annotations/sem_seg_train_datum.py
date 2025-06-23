import os
import sys
from src.otx.engine import Engine
import time
import cv2
PATH_TO_REPO = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/"

sys.path.append(os.path.join(PATH_TO_REPO, "src"))

data_root = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/holesingear-dataset-voc-cleaned_coco"
recipe = os.path.join(PATH_TO_REPO, "src/otx/recipe/semantic_segmentation/litehrnet_18.yaml")

engine = Engine(data_root=data_root,
                work_dir="otx-workspace",
                # task="semantic_segmentation",
                recipe=recipe
                )

engine.train()
results = engine.test()

images_dir = os.path.join(data_root, "train","images")
images_list = os.listdir(images_dir)
for image_path in images_list:
    image_path = os.path.join(images_dir, image_path)
    tic = time.time()
    image = cv2.imread(image_path)
    results = engine.model(image)
    toc = time.time()
    print(f"Time taken for inference: {toc-tic} seconds")



results
