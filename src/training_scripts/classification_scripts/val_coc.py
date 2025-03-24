from pycocotools.coco import COCO
import os
coco_dir = "/home/rgangire/workspace/data/MultiLabel/multilabel_modified/Converted/COCO"
ann_file = os.path.join(coco_dir,"annotations","train.json")
coco = COCO(ann_file)

# Check categories, images, and annotations
print(f"Categories: {len(coco.dataset['categories'])}")
print(f"Images: {len(coco.dataset['images'])}")
print(f"Annotations: {len(coco.dataset['annotations'])}")