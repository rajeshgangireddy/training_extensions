import os
import shutil
import random


def create_folders(output_dir):
    """Creates train, val, test folders with images and masks subdirectories."""
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "masks"), exist_ok=True)


def split_dataset(raw_dir, output_dir, split_ratio=(0.7, 0.2, 0.1), max_size=None, max_per_subset=None, seed=42):
    """
    Splits dataset into train, val, test folders.

    Args:
        raw_dir (str): Path to raw dataset (should contain 'Image' and 'Mask' folders).
        output_dir (str): Path to output dataset.
        split_ratio (tuple): (train_ratio, val_ratio, test_ratio) - should sum to 1.
        max_size (int, optional): Maximum number of image-mask pairs to use.
        max_per_subset (int, optional): Maximum number of images per subset (train, val, test).
        seed (int): Random seed for reproducibility.
    """
    random.seed(seed)

    image_dir = os.path.join(raw_dir, "Image")
    mask_dir = os.path.join(raw_dir, "Mask")

    # Ensure the raw dataset folders exist
    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise ValueError("Raw dataset folders 'Image' and 'Mask' must exist.")

    # Get all image files (assuming masks have the same names)
    images = sorted([f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))])

    if max_size:
        images = images[:max_size]  # Limit dataset size

    random.shuffle(images)  # Shuffle before splitting

    train_size = int(len(images) * split_ratio[0])
    val_size = int(len(images) * split_ratio[1])

    train_files = images[:train_size]
    val_files = images[train_size:train_size + val_size]
    test_files = images[train_size + val_size:]

    # subsitiute for max_per subset if provide for train val or test
    if max_per_subset:
        train_files = train_files[:max_per_subset[0]] if max_per_subset[0] else train_files
    if max_per_subset:
        val_files = val_files[:max_per_subset[1]] if max_per_subset[1] else val_files
    if max_per_subset:
        test_files = test_files[:max_per_subset[2]] if max_per_subset[2] else test_files


    # Create the necessary directories
    create_folders(output_dir)

    def move_files(files, split):
        """Moves image-mask pairs to the respective split folder."""
        for file in files:
            shutil.copy(os.path.join(image_dir, file), os.path.join(output_dir, split, "images", file))
            # masks are with extension png
            mask_file = file.split(".")[0] + ".png"
            shutil.copy(os.path.join(mask_dir, mask_file), os.path.join(output_dir, split, "masks", mask_file))

    # Move files to respective folders
    move_files(train_files, "train")
    move_files(val_files, "val")
    move_files(test_files, "test")

    print(f"Dataset split completed!\nTrain: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    print(f"Saved to: {output_dir}")


# Example usage:
raw_dataset_dir = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/flood/FloodSegmentation/RAW"
output_dir = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/flood_segmetation--es"
split_dataset(raw_dataset_dir, output_dir, split_ratio=(0.6, 0.2, 0.2), max_size=80,max_per_subset=None)