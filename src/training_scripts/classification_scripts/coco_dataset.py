import os
import json
import pandas as pd
import shutil
from tqdm import tqdm
from zipfile import ZipFile
from PIL import Image


def create_coco_format(dataset_dir, csv_file, output_dir, max_images=150):
    # Paths
    images_dir = os.path.join(dataset_dir, "images")
    csv_path = os.path.join(dataset_dir, csv_file)
    output_images_dir = os.path.join(output_dir, "images")
    output_file = os.path.join(output_dir, "annotations", "train.json")
    zip_file = os.path.join(os.path.basename(output_dir), "coco_dataset.zip")
    read_me_file = os.path.join(output_dir, "README.txt")
    # Create output directories
    if os.path.exists(output_images_dir):
        shutil.rmtree(output_images_dir)
    os.makedirs(output_images_dir, exist_ok=True)

    if os.path.exists(output_file):
        os.remove(output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)


    # dummy readme file
    with open(read_me_file, "w") as f:
        f.write("This dataset is created using the COCO format for multi-label classification.\n"
                "The dataset contains images and annotations for training a multi-label classification model.\n"
                "The annotations are in the form of a JSON file in COCO format.\n"
                "The images are stored in the 'images' folder.\n"
                "The annotations are stored in the 'annotations' folder.\n"
                "The dataset is zipped for easy download and use.\n"
                "The dataset is created by Your Name.\n"
                "Date: 2025-01-11\n"
                "Version: 1.0\n"
                "License: MIT\n"
                "Contact:"
                )



    # Read the CSV file
    df = pd.read_csv(csv_path)

    # COCO dataset structure
    coco_format = {
        "info": {
            "description": "Multi-label Classification Dataset",
            "version": "1.0",
            "year": 2025,
            "contributor": "Your Name",
            "date_created": "2025-01-11"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Create categories (unique labels)
    all_labels = set()
    for labels in df["Classes"].dropna():
        all_labels.update(labels.split())
    categories = [{"id": i + 1, "name": label, "supercategory": "none"} for i, label in enumerate(sorted(all_labels))]
    coco_format["categories"] = categories

    # Map category names to IDs
    label_to_id = {category["name"]: category["id"] for category in categories}

    # Add images and annotations
    annotation_id = 1
    image_count = 0
    for image_id, row in tqdm(enumerate(df.itertuples(), start=1), total=len(df), desc="Processing images"):
        if image_count >= max_images:
            break

        image_name = row.Image_Name
        image_path = os.path.join(images_dir, image_name)

        # Get image dimensions
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_name} not found in {images_dir}. Skipping.")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        # Check if the image has at least two labels
        labels = row.Classes.split()
        if len(labels) < 2:
            continue

        # Add image metadata
        coco_format["images"].append({
            "id": image_id,
            "file_name": image_name,
            "width": width,
            "height": height
        })

        # Add annotations for the image
        for label in labels:
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": label_to_id[label],
                "bbox": [],  # Empty for classification
                "area": 0,  # Empty for classification
                "segmentation": [],  # Empty for classification
                "iscrowd": 0
            })
            annotation_id += 1

        # Copy image to output folder
        shutil.copy(image_path, output_images_dir)
        image_count += 1

    # Save COCO JSON to file
    with open(output_file, "w") as f:
        json.dump(coco_format, f, indent=2)

    # # Zip the dataset
    # with ZipFile(zip_file, 'w') as zipf:
    #     for root, _, files in os.walk(output_dir):
    #         for file in files:
    #             file_path = os.path.join(root, file)
    #             zipf.write(file_path, os.path.relpath(file_path, output_dir))

    print(f"COCO dataset saved to {output_file}")
    print(f"Images copied to {output_images_dir}")
    # print(f"Dataset zipped to {zip_file}")


# Example usage
dataset_directory = "/home/rgangire/workspace/data/MultiLabel/multilabel_modified"
csv_filename = "multilabel_classification_fixe.csv"
output_dir = "/home/rgangire/workspace/data/MultiLabel/multilabel_modified/Converted/COCO"
max_images = 150

create_coco_format(dataset_directory, csv_filename, output_dir, max_images)
