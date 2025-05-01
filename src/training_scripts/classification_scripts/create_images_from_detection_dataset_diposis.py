import os
import shutil

# Folders to process
data_root = "/home/rgangire/workspace/datasets/Classification/Processed/Diopsis/images"
data_subsets = ['test', 'train', 'val']
data_folders = [os.path.join(data_root, subset) for subset in data_subsets]
all_images_folder = "/home/rgangire/workspace/datasets/Classification/Processed/Diopsis/crops"


for folder in data_folders:
    count = 0
    for root, _, files in os.walk(folder):
        for fname in files:
            file_path = os.path.join(root, fname)

            # Check if it's a symlink
            if os.path.islink(file_path):
                # Resolve symlink target
                real_path = os.readlink(file_path)

                # Absolute path of source file
                source_path = os.path.join(all_images_folder, os.path.basename(real_path))

                # Remove the symlink
                os.unlink(file_path)

                # Copy the real file
                shutil.copy2(source_path, file_path)

                print(f"Replaced symlink {file_path} with real file from {source_path}")
                count += 1

    print(f"Processed {count} symlinks in {folder}")

