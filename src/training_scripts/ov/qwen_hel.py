import numpy as np
import openvino as ov
import openvino_genai
from PIL import Image
import time

# Choose GPU instead of CPU in the line below to run the model on Intel integrated or discrete GPU

tic = time.time()
pipe = openvino_genai.VLMPipeline("/home/rgangire/Qwen2-VL-2B-Instruct-ov-int4", "CPU")
print(f"Pipeline load : {time.time() - tic:.2f} seconds")

tic = time.time()
image_path = "/home/rgangire/workspace/datasets/Buffer/ButterFlyPap.jpg"
image = Image.open(image_path)
image_data = np.array(image.getdata()).reshape(1, image.size[1], image.size[0], 3).astype(np.uint8)
image_data = ov.Tensor(image_data)
print(f"Image load : {time.time() - tic:.2f} seconds")

prompt = "Can you describe the image?"
tic = time.time()
result = pipe.generate(prompt, image=image_data, max_new_tokens=100)
print(result.texts[0])
print(f"Inference time : {time.time() - tic:.2f} seconds")


prompt = "Can you tell what is in the image?"
tic = time.time()
result = pipe.generate(prompt, image=image_data, max_new_tokens=100)
print(result.texts[0])
print(f"Inference time : {time.time() - tic:.2f} seconds")

prompt = "What's the species of the butterfly in the image?"
tic = time.time()
result = pipe.generate(prompt, image=image_data, max_new_tokens=100)
print(result.texts[0])
print(f"Inference time : {time.time() - tic:.2f} seconds")