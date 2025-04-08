import argparse
from argparse import Namespace

import numpy as np
from PIL import Image

from model_api.models import Model
from model_api.visualizer import Visualizer


def main(args: Namespace):
    image = Image.open(args.image)

    model = Model.create_model(args.model)
    labels = model.labels

    predictions = model(np.array(image))
    pred_probabilities = predictions.raw_scores

    print(f"Input : {args.image}")
    for i, label in enumerate(labels):
        print(f"{label}: {pred_probabilities[i]:.4f} ")

    visualizer = Visualizer()

    if args.output:
        visualizer.save(image=image, result=predictions, path=args.output)
    else:
        visualizer.show(image=image, result=predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="Drilling.jpg")
    parser.add_argument("--model", type=str, default= "model/model.xml")
    parser.add_argument("--output", type=str, required=False)
    args = parser.parse_args()
    main(args)