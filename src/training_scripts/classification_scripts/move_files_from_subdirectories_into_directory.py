import os
import shutil
from tqdm import tqdm
from typing import List, Tuple

class FileMover:
    def __init__(self, source_dir: str, dest_dir: str, operation: str = "move"):
        self.source_dir = source_dir
        self.dest_dir = dest_dir
        self.operation = operation
        self.overwritten_files = []

    def move_files(self) -> None:
        all_files = self._get_all_files(self.source_dir)
        print(f"Total files found: {len(all_files)}")
        self._move_files_to_dest(all_files)
        self._print_overwritten_files()

    def _get_all_files(self, directory: str) -> List[str]:
        files_list = []
        for root, _, files in os.walk(directory):
            for file in files:
                files_list.append(os.path.join(root, file))
        return files_list

    def _move_files_to_dest(self, files: List[str]) -> None:
        os.makedirs(self.dest_dir, exist_ok=True)
        for file in tqdm(files, desc="Moving files"):
            dest_file = os.path.join(self.dest_dir, os.path.basename(file))
            if os.path.exists(dest_file):
                self.overwritten_files.append(dest_file)
            if self.operation == "move":
                shutil.move(file, dest_file)
            elif self.operation == "copy":
                shutil.copy2(file, dest_file)

    def _print_overwritten_files(self) -> None:
        if self.overwritten_files:
            print("Overwritten files:")
            for file in self.overwritten_files:
                print(file)
            print(f"Total overwritten files: {len(self.overwritten_files)}")
        else:
            print("No files were overwritten.")


if __name__ == "__main__":
    source_directory = "/home/rgangire/workspace/datasets/Classification/RAW/AID_ML/AsIs"
    destination_directory = "/home/rgangire/workspace/datasets/Classification/RAW/AID_ML/All"
    operation = "copy"  # Change to "copy" if you want to copy files instead of moving
    file_mover = FileMover(source_directory, destination_directory, operation)
    file_mover.move_files()