import os

def rename_files_in_directory(base_dir):
    """
    Renames all files in the given directory and its subdirectories to a smaller format:
    folder_name_filenumber (6-character zero-padded number).

    Args:
        base_dir (str): The base directory to start renaming files.
    """
    for root, _, files in os.walk(base_dir):
        folder_name = os.path.basename(root)
        for idx, file in enumerate(files, start=1):
            file_ext = os.path.splitext(file)[1]  # Get file extension
            new_name = f"{folder_name}_{str(idx).zfill(6)}{file_ext}"  # Generate new name
            old_path = os.path.join(root, file)
            new_path = os.path.join(root, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

# Example usage
if __name__ == "__main__":
    base_directory = "/home/rgangire/workspace/datasets/Classification/RAW/PlantVillage/PlantVillage-Dataset-master/raw/color"  # Replace with your directory path
    rename_files_in_directory(base_directory)