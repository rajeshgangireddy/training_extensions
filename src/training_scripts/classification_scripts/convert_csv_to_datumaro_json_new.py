import json
import pandas as pd
from typing import List, Dict


def load_csv(file_path: str) -> pd.DataFrame:
    """Load the CSV file into a pandas DataFrame and rename columns if needed."""
    df = pd.read_csv(file_path)
    df.rename(columns={df.columns[0]: "image_id"}, inplace=True)  # Rename first column
    return df


def generate_categories(labels: List[str]) -> Dict:
    """Generate categories from unique labels."""
    label_groups = [
        {"name": f"___{label}", "group_type": "exclusive", "labels": [label]}
        for label in labels
    ]

    label_mappings = {label: idx for idx, label in enumerate(labels)}

    return {
        "label": {
            "label_groups": label_groups,
            "labels": [{"name": label, "parent": "", "attributes": []} for label in labels],
            "attributes": []
        }
    }, label_mappings


def generate_items(df: pd.DataFrame, label_mappings: Dict[str, int]) -> List[Dict]:
    """Generate items for JSON output."""
    items = []
    annotation_id = 0

    for _, row in df.iterrows():
        image_id = row['image_id']
        labels = [label for label in label_mappings.keys() if row[label] == 1]  # Select labels with value 1

        annotations = [
            {"id": annotation_id + i,
             "type": "label",
             "group": label_mappings[label],
             "label_id": label_mappings[label]}
            for i, label in enumerate(labels)
        ]
        annotation_id += len(labels)

        items.append({
            "id": image_id,
            "annotations": annotations,
            "image": {"path": f"{image_id}.jpg"}
        })

    return items


def convert_csv_to_json(csv_path: str, json_path: str):
    """Convert CSV file to JSON format."""
    df = load_csv(csv_path)
    unique_labels = list(df.columns[1:])  # All columns except 'image_id' are labels
    categories, label_mappings = generate_categories(unique_labels)
    items = generate_items(df, label_mappings)

    json_data = {
        "info": {},
        "categories": categories,
        "items": items
    }

    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)


if __name__ == "__main__":
    csv_file_path = '/home/rgangire/workspace/datasets/Classification/RAW/AID_ML/multilabel.csv'
    json_file_path = '/home/rgangire/workspace/datasets/Classification/RAW/AID_ML/multilabel.json'
    convert_csv_to_json(csv_path=csv_file_path, json_path=json_file_path)
    print(f"CSV file successfully converted to JSON format at {json_file_path}")