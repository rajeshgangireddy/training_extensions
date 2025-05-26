import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image
import time


class ModelLoader:
    """Handles loading and managing the OpenVINO model."""
    def __init__(self, model_path: str, device: str = "CPU"):
        self.model_path = model_path
        self.device = device
        self.pipeline = None

    def load_model(self):
        """Loads the model pipeline."""
        if not self.pipeline:
            tic = time.time()
            self.pipeline = openvino_genai.VLMPipeline(self.model_path, self.device)
            print(f"Pipeline loaded in {time.time() - tic:.2f} seconds")
        return self.pipeline


class ImageProcessor:
    """Handles image loading and preprocessing."""
    @staticmethod
    def load_image(image_path: str) -> ov.Tensor:
        """Loads and preprocesses an image."""
        tic = time.time()
        image = Image.open(image_path).convert("RGB")
        image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
        print(f"Image loaded in {time.time() - tic:.2f} seconds")
        return ov.Tensor(image_data)


class InferenceRunner:
    """Handles running inference on the model."""
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def generate_response(self, prompt: str, image_data: ov.Tensor, max_new_tokens: int = 100):
        """Generates a response for a given prompt and image."""
        tic = time.time()
        result = self.pipeline.generate(prompt, image=image_data, max_new_tokens=max_new_tokens)
        print(result.texts[0])
        print(f"Inference time: {time.time() - tic:.2f} seconds")


def main():
    model_path = "/home/rgangire/Qwen2-VL-2B-Instruct-ov-int4"
    image_path = "/home/rgangire/workspace/datasets/Buffer/ButterFlyPap.jpg"

    # Load model
    model_loader = ModelLoader(model_path)
    pipeline = model_loader.load_model()

    # Process image
    image_processor = ImageProcessor()
    image_data = image_processor.load_image(image_path)

    # Run inference
    inference_runner = InferenceRunner(pipeline)

    prompts = [
        "Can you describe the image?",
        "What's the species of the butterfly in the image?",
        "Where can you find butterflies like this?",
        "Can you eat it?",
    ]

    for prompt in prompts:
        inference_runner.generate_response(prompt, image_data)


if __name__ == "__main__":
    main()