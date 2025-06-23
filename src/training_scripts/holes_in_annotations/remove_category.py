import os
import json
import shutil

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def save_annotations(data, out_path):
    with open(out_path, 'w') as f:
        json.dump(data, f, indent=2)

def main(src_root, dst_root, split="default"):
    images_dir = os.path.join(src_root, "images", split)
    ann_json_path = os.path.join(src_root, "annotations", f"instances_{split}.json")
    dst_images_dir = os.path.join(dst_root, "images", split)
    dst_ann_dir = os.path.join(dst_root, "annotations")
    os.makedirs(dst_images_dir, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)

    data = load_annotations(ann_json_path)
    categories = data['categories']

    print("Categories:")
    for idx, cat in enumerate(categories):
        print(f"{idx}: {cat['name']} (id: {cat['id']})")
    remove_idx = int(input("Enter the index of the category to remove: "))
    remove_cat_id = categories[remove_idx]['id']

    # Remove the category from categories list
    new_categories = [cat for cat in categories if cat['id'] != remove_cat_id]

    # Remove annotations for the selected category
    new_annotations = [ann for ann in data['annotations'] if ann['category_id'] != remove_cat_id]

    # Keep only images that have at least one annotation left
    ann_image_ids = set(ann['image_id'] for ann in new_annotations)
    new_images = [img for img in data['images'] if img['id'] in ann_image_ids]

    # Copy images to new location
    for img in new_images:
        src_img_path = os.path.join(images_dir, img['file_name'])
        dst_img_path = os.path.join(dst_images_dir, img['file_name'])
        os.makedirs(os.path.dirname(dst_img_path), exist_ok=True)
        shutil.copy2(src_img_path, dst_img_path)

    # Save new annotation file
    new_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": new_categories
    }
    dst_ann_json = os.path.join(dst_ann_dir, f"instances_{split}.json")
    save_annotations(new_data, dst_ann_json)
    print(f"New dataset saved to {dst_root}")

if __name__ == "__main__":

    src_root = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/holesingear-dataset-coco"
    dst_root = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/holesingear-dataset-coco-cleaned"
    main(src_root, dst_root)