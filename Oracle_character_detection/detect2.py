import torch
from torchvision import transforms
from PIL import Image, ImageDraw
from model import EAST
import os
import numpy as np
import lanms
from dataset import get_rotate_mat

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def resize_img(img):
    '''Resize image to be divisible by 32'''
    w, h = img.size
    # Ensure the width and height are at least divisible by 32
    resize_w = max(32, w if w % 32 == 0 else int(w / 32) * 32)
    resize_h = max(32, h if h % 32 == 0 else int(h / 32) * 32)
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_w = resize_w / w
    ratio_h = resize_h / h
    return img, ratio_h, ratio_w


def load_pil(img):
    '''Convert PIL Image to torch.Tensor'''
    if img.mode != 'RGB':
        img = img.convert('RGB')
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    return t(img).unsqueeze(0)  # Add batch dimension


def is_valid_poly(res, score_shape, scale):
    '''Check if the poly is within the image boundaries'''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False


def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''Restore polygons from feature maps at given positions'''
    polys = []
    valid_pos *= scale
    d = valid_geo[:4, :]
    angle = valid_geo[4, :]

    for i in range(valid_pos.shape[0]):
        x = valid_pos[i, 0]
        y = valid_pos[i, 1]
        y_min = y - d[0, i]
        y_max = y + d[1, i]
        x_min = x - d[2, i]
        x_max = x + d[3, i]
        rotate_mat = get_rotate_mat(-angle[i])
        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y
        coordinates = np.concatenate((temp_x, temp_y), axis=0)
        res = np.dot(rotate_mat, coordinates)
        res[0, :] += x
        res[1, :] += y
        if is_valid_poly(res, score_shape, scale):
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys)


def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2):
    '''Get boxes from feature map'''
    score = score[0, :, :]
    xy_text = np.argwhere(score > score_thresh)
    if xy_text.size == 0:
        return None
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    valid_pos = xy_text[:, ::-1].copy()
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]
    polys_restored = restore_polys(valid_pos, valid_geo, score.shape)
    if polys_restored.size == 0:
        return None

    xy_text = xy_text[:polys_restored.shape[0]]
    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = polys_restored
    boxes[:, 8] = score[xy_text[:, 0], xy_text[:, 1]]
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)
    return boxes


def adjust_ratio(boxes, ratio_w, ratio_h):
    '''Refine boxes according to the original image ratio'''
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)


def detect(img, model, device):
    '''Detect text regions in the image'''
    img, ratio_h, ratio_w = resize_img(img)
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
    return adjust_ratio(boxes, ratio_w, ratio_h)


def plot_boxes(img, boxes):
    '''Plot boxes on the image'''
    if boxes is None:
        return img
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline="green")
    return img


def save_boxes_to_file(boxes, save_path):
    '''Save boxes to a file'''
    with open(save_path, 'w') as f:
        for box in boxes:
            line = ','.join([str(int(b)) for b in box[:-1]]) + '\n'
            f.write(line)


def process_directory(input_dir, output_img_dir, output_boxes_dir, model, device):
    '''Process all images in a directory and save results to specified output directories'''
    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)
    if not os.path.exists(output_boxes_dir):
        os.makedirs(output_boxes_dir)

    img_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        img = Image.open(img_path)

        # Detect text regions in the image
        boxes = detect(img, model, device)

        # Plot the boxes on the image and save it
        plot_img = plot_boxes(img, boxes)
        output_img_path = os.path.join(output_img_dir, img_file)
        plot_img.save(output_img_path)

        # Save the boxes information to a file
        if boxes is not None:
            output_boxes_path = os.path.join(output_boxes_dir,
                                             img_file.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg',
                                                                                                              '.txt'))
            save_boxes_to_file(boxes, output_boxes_path)
            print(f"Processed {img_file}, saved result to {output_img_path} and boxes to {output_boxes_path}")
        else:
            print(f"No boxes detected in {img_file}")


if __name__ == '__main__':
    input_dir = 'test_data'  # Input image folder path
    output_img_dir = 'test_data_result'  # Folder path to save images with detection boxes
    output_boxes_dir = 'test_data_tag'  # Folder path to save detected box information
    model_path = 'model_epoch_600.pth'  # Path to the model weights file

    # Load the model
    model = EAST().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()

    # Process the directory
    process_directory(input_dir, output_img_dir, output_boxes_dir, model, device)
