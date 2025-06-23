import os
import shutil
from PIL import Image
import numpy as np


def parse_labelmap(labelmap_path):
    # Returns header (first line) and list of dictionaries with name and color tuple
    with open(labelmap_path, 'r') as f:
        lines = f.readlines()
    header = lines[0]
    categories = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(':')
        label = parts[0]
        color_str = parts[1]
        color = tuple(map(int, color_str.split(',')))
        categories.append({'name': label, 'color': color})
    return header, categories


def write_labelmap(header, categories, labelmap_path):
    # Writes the header and remaining categories in the same format
    with open(labelmap_path, 'w') as f:
        f.write(header)
        for cat in categories:
            color_str = ','.join(map(str, cat['color']))
            f.write(f"{cat['name']}:{color_str}::\n")


def process_mask(mask_path, remove_color, bg_color, out_path):
    # Loads the mask as an RGB image and converts it to a numpy array.
    # Finds pixels that match the removed category's color and sets them to background.
    img = Image.open(mask_path).convert('RGB')
    mask_img = np.array(img)
    # Create a boolean mask where all three channels match the remove_color
    remove_mask = np.all(mask_img == np.array(remove_color), axis=-1)
    mask_img[remove_mask] = bg_color
    Image.fromarray(mask_img).save(out_path)


def main(src_root, dst_root):
    masks_dir = os.path.join(src_root, "SegmentationClass")
    images_dir = os.path.join(src_root, "JPEGImages")
    labelmap_path = os.path.join(src_root, "labelmap.txt")
    dst_masks_dir = os.path.join(dst_root, "SegmentationClass")
    dst_images_dir = os.path.join(dst_root, "JPEGImages")
    dst_labelmap_path = os.path.join(dst_root, "labelmap.txt")
    os.makedirs(dst_masks_dir, exist_ok=True)
    os.makedirs(dst_images_dir, exist_ok=True)

    header, categories = parse_labelmap(labelmap_path)
    print("Categories:")
    for idx, cat in enumerate(categories):
        print(f"{idx}: {cat['name']} with color {cat['color']}")
    remove_idx = int(input("Enter the index of the category to remove: "))
    remove_color = categories[remove_idx]['color']
    bg_color = categories[0]['color']  # assume background is always the first category

    # Create a new labelmap excluding the removed category.
    new_categories = [cat for i, cat in enumerate(categories) if i != remove_idx]
    write_labelmap(header, new_categories, dst_labelmap_path)

    # Process each mask and copy corresponding image.
    for fname in os.listdir(masks_dir):
        mask_path = os.path.join(masks_dir, fname)
        out_mask_path = os.path.join(dst_masks_dir, fname)
        process_mask(mask_path, remove_color, bg_color, out_mask_path)
        # Get corresponding image name based on the mask name.
        base = os.path.splitext(fname)[0] + ".jpg"
        img_path = os.path.join(images_dir, base)
        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(dst_images_dir, base))

    print(f"New dataset saved to {dst_root}")


if __name__ == "__main__":
    src_root = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/holesingear-dataset-voc"
    dst_root = "/home/rgangire/workspace/datasets/SemanticSegmentation/semantic_seg/holesingear-dataset-voc-cleaned"
    main(src_root, dst_root)
