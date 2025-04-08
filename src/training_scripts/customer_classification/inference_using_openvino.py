import argparse
import openvino as ov
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def get_labels_from_xml(xml_file):
    """
    Get a list of labels of a classification task from the model file.
    The labels are in the order of classification output.
    Args:
        xml_file: the OpenVINO model file

    Returns:
        labels: a list of labels
    """
    xml_content = open(xml_file, 'r').read()
    root = ET.fromstring(xml_content)
    labels = root.find('.//model_info/labels').attrib['value']
    return labels.split()

def main(args):
    labels = get_labels_from_xml(args.model_path)

    core = ov.Core()
    compiled_model = core.compile_model(args.model_path, "AUTO")
    infer_request = compiled_model.create_infer_request()

    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)

    H, W = 224, 224

    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not load image from {args.image_path}")
    image = cv2.resize(image, (W, H))
    image = image[np.newaxis, :]  # Add batch dimension
    image = image.astype(np.float32)  # or np.uint8 depending on model

    # Run inference
    pred_probs = compiled_model([image])["raw_scores"][0]

    # Post-process results
    print("-" * 30)
    print(f"File name: {args.image_path}")
    for i, label in enumerate(labels):
        print(f"{label}: {pred_probs[i]:.4f}")
    print("-" * 30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on an image using OpenVINO model')
    parser.add_argument('--image_path',  help='Path to the input image',default="Travelling.jpg")
    parser.add_argument('--model_path', help='Path to the OpenVINO model XML file', default="model/model.xml")
    args = parser.parse_args()

    main(args)