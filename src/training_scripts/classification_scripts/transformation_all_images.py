import os
import shutil
from PIL import Image
from pathlib import Path


def resize_image(image_path, output_path, size):
    """
    Resize an image to the specified size and save it.
    """
    try:
        with Image.open(image_path) as img:
            img = img.resize(size, Image.ANTIALIAS)
            img.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")


def process_images(input_dir, output_dir=None, size=(256, 256), overwrite=False):
    """
    Recursively resize all images in input_dir and save them in output_dir.
    """
    if output_dir is None:
        if overwrite:
            output_dir = input_dir  # Overwrite images in place
        else:
            output_dir = f"{input_dir}_resized"

    if not overwrite and os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        save_path = os.path.join(output_dir, relative_path)
        os.makedirs(save_path, exist_ok=True)

        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(save_path, file)
                resize_image(input_file_path, output_file_path, size)

    print(f"Processing complete. Images saved to: {output_dir}")


if __name__ == "__main__":
    input_directory = "/path/to/input/directory"  # Change this
    IMAGE_SIZE = 256
    output_directory = f"{input_directory}_{IMAGE_SIZE}"
    resize_size = (IMAGE_SIZE, IMAGE_SIZE)
    overwrite_existing = False

    process_images(input_directory, output_directory, resize_size, overwrite_existing)