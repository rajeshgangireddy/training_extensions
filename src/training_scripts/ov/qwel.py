import huggingface_hub as hf_hub
from openvino import Tensor
from PIL import Image
import openvino_genai as ov_genai
import numpy as np
from pathlib import Path

model_id = "OpenVINO/qwen2.5-7b-instruct-int8-ov"
model_path = "/home/rgangire/qwen2.5-7b-instruct-int8-ov"



def streamer(subword: str) -> bool:
    print(subword, end='', flush=True)

def read_image(path: str) -> Tensor:
    pic = Image.open(path).convert("RGB")
    image_data = np.array(pic.getdata()).reshape(1, pic.size[1], pic.size[0], 3).astype(np.uint8)
    return Tensor(image_data)


def read_images(path: str) -> list[Tensor]:
    entry = Path(path)
    if entry.is_dir():
        return [read_image(str(file)) for file in sorted(entry.iterdir())]
    return [read_image(path)]

if __name__ == "__main__":

    device = "CPU"
    hf_hub.snapshot_download(model_id, local_dir=model_path)
    pipe = ov_genai.VLMPipeline(model_path, device)

    image_dir = "/home/rgangire/workspace/datasets/Buffer"
    rgbs = read_images(image_dir)


    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100

    pipe.start_chat()
    prompt = input('question:\n')
    pipe.generate(prompt, images=rgbs, generation_config=config, streamer=streamer)
    while True:
        try:
            prompt = input("\n----------\n"
                           "question:\n")
        except EOFError:
            break
        pipe.generate(prompt, generation_config=config, streamer=streamer)
    pipe.finish_chat()