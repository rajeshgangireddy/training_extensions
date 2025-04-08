import argparse
import cv2
import numpy as np
from openvino.runtime import Core

def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on an image using OpenVINO model')
    parser.add_argument('-m', '--model', required=True, 
                       help='Path to the OpenVINO model XML file')
    parser.add_argument('-i', '--image', required=True,
                       help='Path to the input image')
    parser.add_argument('-d', '--device', default='CPU',
                       help='Target device for inference (default: CPU)')
    parser.add_argument('--height', type=int, default=224,
                       help='Input height for resizing (default: 224)')
    parser.add_argument('--width', type=int, default=224,
                       help='Input width for resizing (default: 224)')
    parser.add_argument('-l', '--labels', default='Drilling,Travelling',
                       help='Comma-separated list of class labels (e.g., "Travelling,Drilling")')
    parser.add_argument('--rgb', action='store_true',
                       help='Convert input image from BGR to RGB (use if model expects RGB)')
    return parser.parse_args()

def load_model(model_path, device):
    # Initialize the OpenVINO runtime
    core = Core()
    
    # Read the model
    model_xml = model_path
    model = core.read_model(model=model_xml)
    
    # Compile the model for the specified device
    compiled_model = core.compile_model(model=model, device_name=device)
    
    # Get input and output nodes
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    return compiled_model, input_layer, output_layer

def preprocess_image(image_path, height, width, convert_to_rgb=False):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Check image channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Image {image_path} is not a 3-channel color image (shape: {image.shape})")
    
    print(f"Loaded image shape: {image.shape} (assumed BGR format from OpenCV)")
    
    # Convert to RGB if specified
    if convert_to_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print("Converted image to RGB format")
    
    # Resize image to specified size
    resized_image= cv2.resize(image, (width, height))
    
    # Convert to float32 and normalize (assuming model expects 0-1 range)
    # image_processed = resized_image.astype(np.float32)
    # image_processed = image_processed / 255.0




    # Keep in NHWC format (do NOT transpose)
    # Add batch dimension
    image_processed = np.expand_dims(resized_image, axis=0)
    
    return image_processed, image

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Split the labels into a list
    labels = [label.strip() for label in args.labels.split(',')]
    if len(labels) != 2:
        print(f"Warning: Expected 2 labels, got {len(labels)}. Using provided labels: {labels}")
    
    # Load the model
    compiled_model, input_layer, output_layer = load_model(args.model, args.device)
    
    # Preprocess the input image with specified dimensions
    input_data, original_image = preprocess_image(args.image, args.height, args.width, args.rgb)
    
    print(compiled_model([input_data]))

    # Perform inference
    result = compiled_model([input_data])[output_layer]
    
    # Process the output for binary classification
    print(f"File : {args.image}")
    print("Inference completed!")
    print(f"Raw output shape: {result.shape}")
    print(f"Raw output: {result}")
    
    # Handle single-output binary classification
    if result.shape == (1, 1):
        probability = result[0][0]  # Extract the single value
        predicted_class_idx = 1 if probability >= 0.5 else 0
        predicted_label = labels[predicted_class_idx]
        confidence = probability if predicted_class_idx == 1 else 1 - probability
        
        print(f"Predicted label: {predicted_label} (Class {predicted_class_idx})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: {labels[0]}: {(1 - probability):.4f}, {labels[1]}: {probability:.4f}")
    else:
        print("Unexpected output shape. Expected [1, 1] for binary classification with single output.")
        print(f"Raw output: {result}")

if __name__ == '__main__':
    main()