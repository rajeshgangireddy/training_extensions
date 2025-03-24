import os
import random
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
from collections import defaultdict
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


def get_classes(dataset_dir: Path, max_classes: Optional[int] = None) -> List[str]:
    classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
    if max_classes is not None:
        classes = classes[:max_classes]
    return classes


def collect_images(dataset_dir: Path, classes: List[str]) -> dict:
    image_dict = {}
    for class_name in classes:
        class_path = dataset_dir / class_name
        images = sorted([img for img in class_path.iterdir() if img.is_file()])
        image_dict[class_name] = images
    return image_dict


def print_dataset_summary(image_dict: dict, subset_name: str):
    total_images = sum(len(images) for images in image_dict.values())
    print(f"Summary for {subset_name}:")
    class_size = []
    for class_name, images in image_dict.items():
        print(f"  {class_name}: {len(images)} images")
        class_size.append(len(images))
    print(f"Normalized class size: {np.array(class_size) / total_images}")
    print(f"Total images in {subset_name}: {total_images}\n")


def split_images(image_dict: dict, train_size: int, val_size: int, test_size: int, stratified_train_val: bool,
                 uniform_test: bool) -> Tuple[dict, dict, dict]:
    all_class_and_images = [(class_name, img) for class_name, images in image_dict.items() for img in images]
    random.shuffle(all_class_and_images)

    total_size = train_size + val_size + test_size

    if len(all_class_and_images) < total_size:
        print(f"Total images in dataset: {len(all_class_and_images)}")
        print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
        raise ValueError(f"Requested sizes exceed the total number of images in the dataset. Uncomment the line below to adjust sizes.")
        #
        # print(f"Warning: Dataset has only {len(all_images)} images, but {total_size} were requested. Adjusting sizes.")
        # train_size = int((train_size / total_size) * len(all_images))
        # val_size = int((val_size / total_size) * len(all_images))
        # test_size = len(all_images) - (train_size + val_size)

    train_dict, val_dict, test_dict = defaultdict(list), defaultdict(list), defaultdict(list)

    labels = [cls for cls, imgs in all_class_and_images]

    if (not stratified_train_val) and (not uniform_test):
        raise ValueError(" A stratified test set but a uniform train and validation set is not supported."
                         " Please set either stratified_train_val or uniform_test to True.")


    if stratified_train_val:
        train_imgs, val_imgs = train_test_split(all_class_and_images,
                                            train_size=train_size,
                                            test_size=val_size,
                                            stratify=labels
                                            )
    else:
        # UNIFORM SAMPLING
        n_classes = len(image_dict)
        # create an array of len classes to assign number of images per class
        n_images_per_class_train = divide_and_distribute(train_size, n_classes)
        train_imgs = []
        for i, (class_name, imgs) in enumerate(image_dict.items()):
            train_imgs.extend([(class_name, img) for img in imgs[:n_images_per_class_train[i]]])

        # remove images that are already in train
        remaining_images_dict = defaultdict(list)
        for cls_name_image_pair in  all_class_and_images:
            if cls_name_image_pair not in train_imgs:
                class_name, img = cls_name_image_pair
                remaining_images_dict[class_name].append(img)

        print_dataset_summary(remaining_images_dict, "Remaining Images After Train Split")

        val_imgs = []
        n_images_per_class_val = divide_and_distribute(val_size, n_classes)
        for i, (class_name, imgs) in enumerate(remaining_images_dict.items()):
            val_imgs.extend([(class_name, img) for img in imgs[:n_images_per_class_val[i]]])

    for class_name, img in train_imgs:
        train_dict[class_name].append(img)
    for class_name, img in val_imgs:
        val_dict[class_name].append(img)

    # remove the images that are already in train and val
    remaining_images_dict = defaultdict(list)
    for cls_name_image_pair in  all_class_and_images:
        # cls_name_image_pair is a tuple of class name and image path
        if cls_name_image_pair not in train_imgs and cls_name_image_pair not in val_imgs:
            class_name, img = cls_name_image_pair
            remaining_images_dict[class_name].append(img)


    print_dataset_summary(remaining_images_dict, "Remaining Images")

    if uniform_test:
        n_classes = len(remaining_images_dict)
        # create an array of len classes to assign number of images per class
        n_images_per_class = divide_and_distribute(test_size, n_classes)
        for i, (class_name, imgs) in enumerate(remaining_images_dict.items()):
            test_dict[class_name] = imgs[:n_images_per_class[i]]

    else:
        # stratified test split
        remaining_label_images = [(class_name, img) for class_name, images in remaining_images_dict.items() for img in images]
        labels = [cls for cls, imgs in remaining_label_images]
        test_imgs, _ = train_test_split(remaining_label_images,
                                        train_size=test_size,
                                        stratify=labels if stratified_train_val else None,
                                        )
        for class_name, img in test_imgs:
            test_dict[class_name].append(img)

    return train_dict, val_dict, test_dict

def divide_and_distribute(total, parts):
    quotient, remainder = divmod(total, parts)
    distribution = [quotient] * parts
    for i in range(remainder):
        distribution[i] += 1
    return distribution


def resize_and_save(image_path: Path, output_path: Path, size: Tuple[int, int] = (640, 640)):
    image = Image.open(image_path)
    image = image.resize(size, Image.Resampling.BILINEAR)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_split(images_dict: dict, output_dir: Path):
    for class_name, images in images_dict.items():
        for image_path in images:
            output_path = output_dir / class_name / image_path.name
            resize_and_save(image_path, output_path)


def create_splits(dataset_dir: str, output_dir: str, train_size: int, val_size: int, test_size: int,
                  stratified_train_val: bool = True, uniform_test: bool = False, max_classes: Optional[int] = None):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = get_classes(dataset_dir, max_classes)
    image_dict = collect_images(dataset_dir, classes)
    print_dataset_summary(image_dict, "Original Dataset")

    train_dict, val_dict, test_dict = split_images(image_dict, train_size, val_size, test_size, stratified_train_val,
                                                   uniform_test)

    print_dataset_summary(train_dict, "Train Subset")
    print_dataset_summary(val_dict, "Validation Subset")
    print_dataset_summary(test_dict, "Test Subset")

    save_split(train_dict, output_dir / 'train')
    save_split(val_dict, output_dir / 'val')
    save_split(test_dict, output_dir / 'test')

    print(f"Dataset splits created at: {output_dir}")


if __name__ == "__main__":
    # This the script to use for creating a subset of a dataset using the directory structure.
    # The dataset dir should have classes as subdirectories and images in the subdirectories.
    # This is primarily for multi-class classification datasets.


    DATASET_DIR =  "/home/rgangire/workspace/datasets/Classification/RAW/CUB_200_2011/images"
    DATASET_NAME = "CUB"
    OUTPUT_DIR_ROOT = "/home/rgangire/workspace/datasets/Classification/Processed/"
    TRAIN_SIZE = 3764
    VAL_SIZE = 900
    TEST_SIZE = 1200
    STRATIFIED_TRAIN_VAL = True # False = UNIFORM SAMPLING
    UNIFORM_TEST = True # Because we measure accuracy instead of Recall
    MAX_CLASSES = 100 # None for all classes

    suffix = f"{TRAIN_SIZE}_{VAL_SIZE}_{TEST_SIZE}"
    suffix += "_STRAT" if STRATIFIED_TRAIN_VAL else "_UNIFORM"
    suffix += "_UNIFORM-TESTSET" if UNIFORM_TEST else ""
    suffix += f"_NC-{MAX_CLASSES}" if MAX_CLASSES else ""

    OUTPUT_DIR = OUTPUT_DIR_ROOT + f"{DATASET_NAME}_{suffix}"

    create_splits(dataset_dir=DATASET_DIR, output_dir=OUTPUT_DIR, train_size=TRAIN_SIZE, val_size=VAL_SIZE,
                  test_size=TEST_SIZE, stratified_train_val=STRATIFIED_TRAIN_VAL, uniform_test=UNIFORM_TEST,
                  max_classes=MAX_CLASSES)

