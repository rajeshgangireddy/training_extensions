import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image


class MultiLabelDatasetProcessor:
    def __init__(self, csv_file_source, images_source, output_file, output_images_folder, max_images):
        self.csv_file = csv_file_source
        self.images_folder = images_source
        self.output_file = output_file
        self.output_images_folder = output_images_folder
        self.max_images = max_images
        self.label_to_id = {}
        self.datumaro_format = self._initialize_datumaro_format()

        self._prepare_directories()

    def _prepare_directories(self):
        """Ensure output directories are clean and ready."""
        if os.path.exists(self.output_images_folder):
            shutil.rmtree(self.output_images_folder)
        os.makedirs(self.output_images_folder, exist_ok=True)
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

    @staticmethod
    def _initialize_datumaro_format():
        """Returns the base structure of the Datumaro JSON format."""
        return {
            "dm_format_version": "1.0",
            "media_type": 2,
            "infos": {
                "ScExtractorVersion": "1.0",
                "GetiProjectTask": "classification",
                "GetiTaskTypeLabels": []
            },
            "categories": {
                "label": {"labels": [], "label_groups": [], "attributes": []},
                "mask": {"colormap": []}
            },
            "items": []
        }

    def _load_csv(self):
        """Loads the dataset CSV file."""
        return pd.read_csv(self.csv_file, sep=",")

    def _extract_labels(self, df):
        """Extracts unique labels and assigns them unique IDs."""
        unique_labels = set()
        possible_class_columns = ["Classes"]

        # check if any of the possible class columns are
        # present in the dataset
        for column in possible_class_columns:
            if column in df.columns:
                break
        else:
            # extract headers except the first one as class names. These as class names
            class_names = df.columns[1:]
            df["Classes"] = df[class_names].apply(lambda x: " ".join(x.dropna()), axis=1)
            print("Classes column not found. Created a new column 'Classes' by combining all other columns.")

        for labels in df["Classes"].dropna():
            unique_labels.update(labels.split())

        self.label_to_id = {label: idx for idx, label in enumerate(sorted(unique_labels))}

        for label, idx in self.label_to_id.items():
            self.datumaro_format["categories"]["label"]["labels"].append(
                {"name": label, "parent": "", "attributes": []}
            )
            self.datumaro_format["categories"]["label"]["label_groups"].append(
                {"name": f"Classification labels___{label}", "group_type": "exclusive", "labels": [label]}
            )
            self.datumaro_format["categories"]["mask"]["colormap"].append(
                {"label_id": idx, "r": (idx * 50) % 256, "g": (idx * 100) % 256, "b": (idx * 150) % 256}
            )

        self.datumaro_format["infos"]["GetiTaskTypeLabels"].append(["classification", list(sorted(unique_labels))])

    def _process_images(self, df):
        """Processes images and creates annotations."""
        image_count = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):

            if max_images and image_count >= self.max_images:
                break

            image_name = row["Image_Name"]
            labels = row["Classes"] if pd.notna(row["Classes"]) else ""
            label_list = labels.split()

            if len(label_list) < 2:
                continue

            annotations = [
                {"id": idx, "type": "label", "attributes": {}, "group": 0, "label_id": self.label_to_id[label]}
                for idx, label in enumerate(label_list)
            ]

            src_image_path = os.path.join(self.images_folder, image_name)
            dst_image_path = os.path.join(self.output_images_folder, image_name)

            try:
                with Image.open(src_image_path) as img:
                    width, height = img.size
            except FileNotFoundError:
                print(f"Warning: Image not found - {src_image_path}")
                continue

            self.datumaro_format["items"].append({
                "id": os.path.splitext(image_name)[0],
                "annotations": annotations,
                "attr": {"has_empty_label": False},
                "image": {
                    "path": os.path.relpath(dst_image_path, os.path.dirname(self.output_file)),
                    "size": [height, width]
                }
            })

            shutil.copy(src_image_path, dst_image_path)
            image_count += 1

    def _save_json(self):
        """Saves the Datumaro formatted JSON file."""
        with open(self.output_file, "w") as f:
            json.dump(self.datumaro_format, f, indent=4)
        print(f"Datumaro annotations saved to {self.output_file}")

    def run(self):
        """Main function to execute all steps."""
        df = self._load_csv()
        self._extract_labels(df)
        self._process_images(df)
        self._save_json()
        print(f"Images with at least two labels saved to {self.output_images_folder}")


if __name__ == "__main__":
    root_path = "/home/rgangire/workspace/datasets/Classification/RAW/AID_ML"
    csv_file = os.path.join(root_path, "multilabel.csv")
    images_folder = os.path.join(root_path, "All")
    output_file = os.path.join(root_path, "Datumaro", "multilabel.json")
    output_images_folder = os.path.join(root_path, "Datumaro", "images")
    max_images = 150
    processor = MultiLabelDatasetProcessor(
        csv_file_source=csv_file,
        images_source=images_folder,
        output_file=output_file,
        output_images_folder=output_images_folder,
        max_images=max_images
    )
    processor.run()
