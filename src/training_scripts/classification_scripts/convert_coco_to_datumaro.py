from datumaro.components.project import Dataset
import os

# Paths
coco_dir = "/home/rgangire/workspace/data/MultiLabel/multilabel_modified/Converted/COCO"
ann_file = os.path.join(coco_dir, "annotations", "train.json")
datumaro_output_dir = "/home/rgangire/workspace/data/MultiLabel/multilabel_modified/Converted/Datumaro"

# Import COCO dataset
dataset = Dataset.import_from(ann_file, format="coco_instances")

# Export to Datumaro format
os.makedirs(datumaro_output_dir, exist_ok=True)
dataset.export(datumaro_output_dir, format="datumaro")

print(f"COCO dataset successfully converted to Datumaro format at {datumaro_output_dir}")
