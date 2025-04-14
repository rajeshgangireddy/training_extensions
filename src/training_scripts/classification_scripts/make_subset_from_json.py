import json
import os
import shutil
import random
from typing import List, Dict, Tuple


def load_json(json_path: str) -> Dict:
    """Load the JSON file containing annotations."""
    with open(json_path, 'r') as f:
        return json.load(f)


def split_data(items: List[Dict], num_train: int, num_test: int, num_val: int) -> Tuple[List, List, List]:
    """Shuffle and split the dataset into train, test, and validation sets."""
    random.shuffle(items)
    return items[:num_train], items[num_train:num_train + num_test], items[
                                                                     num_train + num_test:num_train + num_test + num_val]


def save_json(data: Dict, output_path: str):
    """Save the JSON data to a file."""

    split_name = output_path.split("/")[-1].split(".")[0]
    summarize_json(data=data,split = split_name)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def copy_images(items: List[Dict], image_dir: str, subset_dir: str):
    """Copy images to their respective directories."""
    os.makedirs(subset_dir, exist_ok=True)
    for item in items:
        src_path = os.path.join(image_dir, os.path.basename(item['image']['path']))
        dest_path = os.path.join(subset_dir, os.path.basename(item['image']['path']))
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)
        else:
            print(f"Warning: Image not found - {src_path}")


def summarize_json(data: dict, split: str):
    """Summarize the content of a JSON file."""

    num_images = len(data.get('items', []))
    num_classes = len(data.get('categories', {}).get('label', {}).get('labels', []))
    num_label_groups = len(data.get('categories', {}).get('label', {}).get('label_groups', []))

    print(f"Split: {split}")
    print(f"Number of images: {num_images}")
    print(f"Number of classes: {num_classes}")
    print(f"Number of label groups: {num_label_groups}")


def create_dataset(json_path: str, image_dir: str, output_dir: str, num_train: int, num_test: int, num_val: int):
    """Create the dataset with train, test, and validation subsets."""
    data = load_json(json_path)
    items = data['items']
    train_items, test_items, val_items = split_data(items, num_train, num_test, num_val)

    # add num_train_val_test to output_dir
    output_dir = output_dir +  f"_{num_train}_{num_val}_{num_test}"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        print(f"Deleted existing directory: {output_dir}")

    # Prepare output directories
    annotations_dir = os.path.join(output_dir, 'annotations')
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Add info field if not present
    if "info" not in data:
        data["info"] = {}

    # Save annotation files
    save_json({"info": data["info"], "categories": data["categories"], "items": train_items},
              os.path.join(annotations_dir, 'train.json'))
    save_json({"info": data["info"], "categories": data["categories"], "items": test_items},
              os.path.join(annotations_dir, 'test.json'))
    save_json({"info": data["info"], "categories": data["categories"], "items": val_items},
              os.path.join(annotations_dir, 'val.json'))

    # Copy images to respective subsets
    copy_images(train_items, image_dir, os.path.join(images_dir, 'train'))
    copy_images(test_items, image_dir, os.path.join(images_dir, 'test'))
    copy_images(val_items, image_dir, os.path.join(images_dir, 'val'))


    # count the number of images in each split
    train_images_copied = os.listdir(os.path.join(images_dir, 'train'))
    test_images_copied  = os.listdir(os.path.join(images_dir, 'test'))
    val_images_copied  = os.listdir(os.path.join(images_dir, 'val'))

    print(f"Number of images Copied (Train/Val/Test): {len(train_images_copied)}/{len(val_images_copied)}/{len(test_images_copied)}")




    print(f"Dataset created successfully in {output_dir}")


def main():
    """Main function to define paths and parameters."""
    json_path = "/home/rgangire/workspace/datasets/Classification/RAW/CUB-3-LEVELS-HLABEL-NoSingleCls-GETI-DATUMARO/annotations/default.json"
    image_dir = "/home/rgangire/workspace/datasets/Classification/RAW/CUB-3-LEVELS-HLABEL-NoSingleCls-GETI-DATUMARO/images/default"
    output_dir = "/home/rgangire/workspace/datasets/Classification/Processed/CUB-H-Label-Small-3L-6N-NoMultilabel"
    num_train,  num_val, num_test = 313, 64, 100

    create_dataset(json_path, image_dir, output_dir, num_train, num_test, num_val)


if __name__ == "__main__":
    main()