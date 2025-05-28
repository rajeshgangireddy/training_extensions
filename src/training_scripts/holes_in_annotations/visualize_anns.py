import os
import json
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_annotations(json_path):
    """Load annotations from a JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_image_annotations(image_id, annotations):
    """Get annotations for a specific image ID."""
    return [ann for ann in annotations if ann['image_id'] == image_id]


def draw_segmentation(image, annotations, categories, show_bboxes=False):
    """Draw segmentation masks and optionally bounding boxes on the image."""
    overlay = image.copy()
    bbox_thickness = 10
    for ann in annotations:
        segmentation = np.array(ann['segmentation'][0], dtype=np.int32).reshape(-1, 2)
        category_id = ann['category_id']
        color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for each category
        cv2.fillPoly(overlay, [segmentation], color)
        cv2.polylines(overlay, [segmentation], isClosed=True, color=(0, 0, 0), thickness=2)
        label = categories[category_id - 1]['name']
        x, y = segmentation[0]
        cv2.putText(overlay, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if show_bboxes and 'bbox' in ann:
            x, y, w, h = map(int, ann['bbox'])
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, bbox_thickness)
    return cv2.addWeighted(image, 0.5, overlay, 0.5, 0)


def main(image_dir, json_path, max_images=None, show_bboxes=False):
    """Main function to display images and annotations, with optional limit and bbox display."""
    annotations_data = load_annotations(json_path)
    images = annotations_data['images']
    annotations = annotations_data['annotations']
    categories = annotations_data['categories']

    shown = 0
    for img_info in images:
        if max_images is not None and shown >= max_images:
            break
        img_path = os.path.join(image_dir, img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_annotations = get_image_annotations(img_info['id'], annotations)
        annotated_image = draw_segmentation(image, img_annotations, categories, show_bboxes=show_bboxes)

        # Display the original and annotated images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("Annotated Image")
        plt.imshow(annotated_image)
        plt.axis("off")

        plt.show()
        shown += 1


if __name__ == "__main__":
    dataset_root = "/home/rgangire/workspace/datasets/instance_seg/wgisd_small/wgisd_small/1/"
    split = "test"
    images_dir = os.path.join(dataset_root, "images", split)
    ann_json_path = os.path.join(dataset_root, "annotations", f"instances_{split}.json")
    max_images = 3  # Set to None to show all images, or set to an integer to limit
    show_bboxes = True  # Set to True to show bounding boxes
    main(images_dir, ann_json_path, max_images, show_bboxes)
