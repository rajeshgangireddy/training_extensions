import os
import shutil
import random


def create_folders(output_dir):
    """Creates train, val, test folders with images and masks subdirectories."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)


def collect_files(root_dir, split):
    """Collects all image-mask file pairs from the raw dataset."""
    img_dir = os.path.join(root_dir, "leftImg8bit", split)
    mask_dir = os.path.join(root_dir, "gtFine", split)

    images = []
    for sub_folder in os.listdir(img_dir):
        sub_img_path = os.path.join(img_dir, sub_folder)
        sub_mask_path = os.path.join(mask_dir, sub_folder)

        if os.path.isdir(sub_img_path) and os.path.isdir(sub_mask_path):
            for img_file in os.listdir(sub_img_path):
                if img_file.endswith(".jpg"):
                    mask_file = img_file.replace("_image.jpg", "_label.png")
                    if os.path.exists(os.path.join(sub_mask_path, mask_file)):
                        images.append((os.path.join(sub_img_path, img_file), os.path.join(sub_mask_path, mask_file)))

    return images


def split_dataset(raw_dir, output_dir, train_split=0.8, max_size=None, seed=42):
    """
    Splits dataset into train, val, test with structured folders.

    Args:
        raw_dir (str): Path to the raw dataset.
        output_dir (str): Path to output dataset.
        train_split (float): Fraction of original train data to keep for training (rest goes to test).
        max_size (int, optional): Maximum dataset size.
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    train_files = collect_files(raw_dir, "train")
    val_files = collect_files(raw_dir, "val")

    # Apply max_size limit
    if max_size:
        train_files = train_files[:max_size]
        val_files = val_files[:max_size]

    # Shuffle and split train into train & test
    random.shuffle(train_files)
    train_size = int(len(train_files) * train_split)
    train_split_files = train_files[:train_size]
    test_split_files = train_files[train_size:]

    # Create output folders
    create_folders(output_dir)

    def move_files(files, split):
        """Moves image-mask pairs to the respective split folder."""
        for img_path, mask_path in files:
            shutil.copy(img_path, os.path.join(output_dir, split, "images", os.path.basename(img_path)))
            # new mask file name is same as image file name but with png extension
            new_mask_file_name = os.path.basename(img_path).replace(".jpg", ".png")
            new_mask_path = os.path.join(output_dir, split, "masks", new_mask_file_name)
            shutil.copy(mask_path, new_mask_path)
            print(f"Moved {os.path.basename(img_path)} to {split}/images")

    # Move files to respective folders
    move_files(train_split_files, "train")
    move_files(test_split_files, "test")
    move_files(val_files, "val")

    print(
        f"Dataset split completed!\nTrain: {len(train_split_files)}, Test: {len(test_split_files)}, Val: {len(val_files)}")
    print(f"Output directory: {output_dir}")


# Example usage:
RAW_DIR = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/IDD_RAW/idd20k_lite"
OUTPUT_DIR = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/IDD_RAW/idd20k_lite_PROCESSED"
split_dataset(RAW_DIR, OUTPUT_DIR, train_split=0.8, max_size=10000)