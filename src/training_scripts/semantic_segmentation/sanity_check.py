import os
import json
import numpy as np
from PIL import Image


def check_and_convert_to_jpg(image_path):
    """Convert an image to JPG if it is not already in JPG format."""
    if not image_path.lower().endswith(".jpg"):
        img = Image.open(image_path)
        new_path = os.path.splitext(image_path)[0] + ".jpg"
        img.convert("RGB").save(new_path, "JPEG")
        os.remove(image_path)
        print(f"Converted {image_path} to {new_path}")
        return os.path.basename(new_path)
    return os.path.basename(image_path)


def check_and_convert_mask_for_classes(mask_path, num_classes):
    """
    Check if a mask's pixel values are within the range [0, num_classes - 1].
    If not, convert them to be within the range.

    Args:
        mask_path (str): Path to the mask image.
        num_classes (int): Number of classes.

    Returns:
        None: The mask is updated in place if necessary.
    """
    mask = Image.open(mask_path)
    mask_array = np.array(mask)

    # Ensure pixel values are within the range [0, num_classes - 1]
    # histogram of mask_array
    # hist, _ = np.histogram(mask_array, bins=np.arange(num_classes + 1))
    if mask_array.max() >= num_classes:
        unique, counts = np.unique(mask_array, return_counts=True)
        print(f"Unique pixel values in {mask_path}: {dict(zip(unique, counts))}")
        print(f"⚠️ Mask {mask_path} has pixel values outside the range [0, {num_classes - 1}]. Updating...")
        mask_array = np.clip(mask_array, 0, num_classes - 1)
        # Convert to uint8 for saving
        mask_array = mask_array.astype(np.uint8)
        updated_mask = Image.fromarray(mask_array)
        updated_mask.save(mask_path)
        print(f"Updated mask saved at {mask_path}")
    return mask_path

def check_dataset_structure(base_path):
    """Checks dataset folder structure, correct image formats, image-mask consistency, and dataset_meta.json validation."""
    required_subsets = ["train", "test", "val"]
    required_dirs = ["images", "masks"]

    for dataset in os.listdir(base_path):
        dataset_path = os.path.join(base_path, dataset)

        if not os.path.isdir(dataset_path):
            continue  # Skip non-directory files

        print(f"\nChecking dataset: {dataset}")

        meta_content = None  # To store content of the first dataset_meta.json file

        # Check subsets exist
        for subset in required_subsets:
            subset_path = os.path.join(dataset_path, subset)
            if not os.path.exists(subset_path):
                print(f"❌ Missing subset folder: {subset_path}")
                continue

            # Check images and masks exist
            for req_dir in required_dirs:
                req_path = os.path.join(subset_path, req_dir)
                if not os.path.exists(req_path):
                    print(f"❌ Missing folder: {req_path}")
                    continue

            # Check dataset_meta.json exists and validate consistency
            meta_file_path = os.path.join(subset_path, "dataset_meta.json")
            if not os.path.exists(meta_file_path):
                print(f"❌ Missing dataset_meta.json in {subset_path}")
            else:
                try:
                    with open(meta_file_path, "r") as f:
                        meta_data = json.load(f)

                    if meta_content is None:
                        meta_content = meta_data  # Store the first meta file content
                    elif meta_content != meta_data:
                        print(f"⚠ Inconsistent dataset_meta.json content in {subset_path}")

                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON format in {meta_file_path}")

            # number of classes according to meta file
            label_map = meta_data["label_map"]
            num_classes = len(label_map)
            # Check images and corresponding masks
            img_dir = os.path.join(subset_path, "images")
            mask_dir = os.path.join(subset_path, "masks")

            if os.path.exists(img_dir) and os.path.exists(mask_dir):

                # If there is a .DS_Store file, remove it
                if ".DS_Store" in os.listdir(img_dir):
                    os.remove(os.path.join(img_dir, ".DS_Store"))
                if ".DS_Store" in os.listdir(mask_dir):
                    os.remove(os.path.join(mask_dir, ".DS_Store"))

                images = os.listdir(img_dir)
                masks = os.listdir(mask_dir)

                # Convert non-jpg images to jpg
                updated_images = []
                for img_file in images:
                    img_path = os.path.join(img_dir, img_file)
                    new_img_name = check_and_convert_to_jpg(img_path)
                    updated_images.append(new_img_name)

                # Validate mask-image consistency and check binary masks
                for img_name in updated_images:
                    mask_name = os.path.splitext(img_name)[0] + ".png"
                    mask_path = os.path.join(mask_dir, mask_name)
                    if mask_name not in masks:
                        print(f"⚠ Missing mask: {mask_name} for image {img_name} in {subset} in dataset {dataset}")
                    else:
                        check_and_convert_mask_for_classes(mask_path,num_classes)

    print("\nSanity check completed!")


DATASET_ROOT_DIR = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/sat"
check_dataset_structure(DATASET_ROOT_DIR)
