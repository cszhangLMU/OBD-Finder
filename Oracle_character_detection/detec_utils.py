import torch
from torchvision import transforms
from PIL import Image
import os
from .model import EAST
from .dataset import get_rotate_mat
import numpy as np
import lanms
from .detect import detect


def process_two_images(img_path1, img_path2, model, device, output_dir):
    """
    Input two image paths and return their annotation files.
    Args:
        img_path1 (str): Path of the first image.
        img_path2 (str): Path of the second image.
        model (torch.nn.Module): Loaded EAST model.
        device (torch.device): The device to be used (CPU or GPU).
        output_dir (str): Path to save the output annotation files.
    """

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def detect_and_save(img_path, model, device, output_dir):
        """Detect on a single image and save the annotation file."""
        img = Image.open(img_path)
        boxes = detect(img, model, device)

        if boxes is None:
            print(f"No boxes detected for {img_path}")
            return

        # Save the annotation file
        output_file = os.path.join(output_dir, os.path.basename(img_path).replace('.png', '.txt'))
        seq = [','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes]
        with open(output_file, 'w') as f:
            f.writelines(seq)
        print(f"Saved annotation for {img_path} at {output_file}")

    # Detect and save annotation for the first image
    detect_and_save(img_path1, model, device, output_dir)

    # Detect and save annotation for the second image
    detect_and_save(img_path2, model, device, output_dir)


if __name__ == '__main__':
    # Image paths
    img_path1 = '/home/dengbinquan/XIAOWU/EAST/data/test_data/2711.png'
    img_path2 = '/home/dengbinquan/XIAOWU/EAST/data/test_data/2715.png'

    # Model path
    model_path = '/home/dengbinquan/XIAOWU/EAST/pths/model_epoch_600.pth'

    # Output directory for annotation files
    output_dir = '/home/dengbinquan/XIAOWU/EAST/data/test_data'

    # Load the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Call the processing function
    process_two_images(img_path1, img_path2, model, device, output_dir)
