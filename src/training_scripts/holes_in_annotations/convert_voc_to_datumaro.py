import datumaro as dm


data_root = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/holesingear-dataset-voc-cleaned"
dataset = dm.Dataset.import_from(data_root, 'voc')

# convert to datumaro format
datumaro_root = data_root + '_datumaro'
coco_root = data_root + '_coco'


dataset.export(datumaro_root, format='datumaro', save_media=True)
dataset.export(coco_root, format='coco',save_media=True)
print(f"Converted dataset saved to {datumaro_root}")
print(f"Converted dataset saved to {coco_root}")

