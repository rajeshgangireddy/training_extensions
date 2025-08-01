import os
from otx.backend.native.engine import OTXEngine
from otx.data.module import OTXDataModule
import datumaro as dm
from otx.types.task import OTXTaskType
from otx.config.data import SubsetConfig
from torchvision.transforms.v2 import Resize,Compose,ToDtype,Normalize
import torch
import pycocotools.mask as mask_utils
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_REPO = "/home/rgangire/workspace/code/repos/Geti-Labs/OTX/repo/training_extensions/"
data_root = "/home/rgangire/workspace/datasets/Collection/holes_in_ann/topgear-dataset-voc"
recipe = os.path.join(PATH_TO_REPO, "src/otx/recipe/semantic_segmentation/litehrnet_18.yaml")

data_export_dir_coco = os.path.join(os.path.dirname(data_root),"exported_data","coco")
data_export_dir_datumaro = os.path.join(os.path.dirname(data_root),"exported_data","datumaro")

dataset = dm.Dataset.import_from(data_root, 'voc')
dataset.export(data_export_dir_coco, 'coco', save_media=True)
dataset.export(data_export_dir_datumaro, 'datumaro', save_media=True)

dataset_coco = dm.Dataset.import_from(data_export_dir_coco, 'coco')
dataset_datum = dm.Dataset.import_from(data_export_dir_datumaro, 'datumaro')


dataset_coco.transform("polygons_to_masks")
# visualise voc item annotations
voc_item = dataset[0]
for ann in voc_item.annotations:
    mask = ann.image
    # show this mask
    if isinstance(mask, np.ndarray):
        plt.imshow(mask)
        plt.title(f"Annotation: {ann.type} - {ann.label}")
        plt.axis('off')
        plt.show()
    else:
        print(f"Annotation {ann.type} does not have an image mask.")


coco_item = dataset_coco[0]
for ann in coco_item.annotations:
    mask = ann.image
    # show this mask
    if isinstance(mask, np.ndarray):
        plt.imshow(mask)
        plt.title(f"Annotation: {ann.type} - {ann.label}")
        plt.axis('off')
        plt.show()
    else:
        print(f"Annotation {ann.type} does not have an image mask.")


transforms = Compose([Resize(size=(448, 448)),ToDtype(dtype=torch.float32)])



train_config=SubsetConfig(batch_size=1,subset_name="train",transforms=transforms)
val_config=SubsetConfig(batch_size=1,subset_name="val",transforms=transforms)
test_config=SubsetConfig(batch_size=1,subset_name="test",transforms=transforms)


datamodule = OTXDataModule(task=OTXTaskType.SEMANTIC_SEGMENTATION,
                           data_format="voc",
                           data_root=data_root,
                           train_subset=train_config,
                           val_subset=val_config,
                            test_subset=test_config,
                           )




from otx.backend.native.engine import OTXEngine

engine = OTXEngine(
    data=data_export_dir_coco,
    model=recipe,
    max_epochs=9,
)
