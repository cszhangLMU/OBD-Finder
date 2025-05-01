import numpy as np
from PIL import Image
import torch

from siamese import Siamese

if __name__ == "__main__":
    # Initialize the model
    model = Siamese(model_path='/home/dengbinquan/XIAOWU/Siamese-pytorch/logs1/best_epoch_weights.pth',
                    input_shape=[105, 105], cuda=True)

    while True:
        image_1 = input('Input image_1 filename (or type "q" to quit):')
        if image_1.lower() == 'q':
            break
        try:
            image_1 = Image.open(image_1)
        except FileNotFoundError:
            print(f"File {image_1} not found! Try again!")
            continue
        except Exception as e:
            print(f"An error occurred while opening {image_1}: {e}")
            continue

        image_2 = input('Input image_2 filename (or type "q" to quit):')
        if image_2.lower() == 'q':
            break
        try:
            image_2 = Image.open(image_2)
        except FileNotFoundError:
            print(f"File {image_2} not found! Try again!")
            continue
        except Exception as e:
            print(f"An error occurred while opening {image_2}: {e}")
            continue

        # Call the model to detect the similarity between the two images
        probability = model.detect_image(image_1, image_2)
        print(f"Similarity between images: {probability.item():.3f}")
