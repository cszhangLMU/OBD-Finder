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
    输入两张图片路径，返回它们的标注文件
    Args:
        img_path1 (str): 第一张图片的路径
        img_path2 (str): 第二张图片的路径
        model (torch.nn.Module): 已加载的EAST模型
        device (torch.device): 使用的设备 (CPU or GPU)
        output_dir (str): 输出的标注文件保存路径
    """
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    def detect_and_save(img_path, model, device, output_dir):
        """对单张图片进行检测并保存标注文件"""
        img = Image.open(img_path)
        boxes = detect(img, model, device)
        
        if boxes is None:
            print(f"No boxes detected for {img_path}")
            return
        
        # 保存标注文件
        output_file = os.path.join(output_dir, os.path.basename(img_path).replace('.png', '.txt'))
        seq = [','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes]
        with open(output_file, 'w') as f:
            f.writelines(seq)
        print(f"Saved annotation for {img_path} at {output_file}")
    
    # 对第一张图片进行检测并保存标注
    detect_and_save(img_path1, model, device, output_dir)
    
    # 对第二张图片进行检测并保存标注
    detect_and_save(img_path2, model, device, output_dir)


if __name__ == '__main__':
    # 图片路径
    img_path1 = '/home/dengbinquan/XIAOWU/EAST/data/test_data/2711.png'
    img_path2 = '/home/dengbinquan/XIAOWU/EAST/data/test_data/2715.png'
    
    # 模型路径
    model_path  = '/home/dengbinquan/XIAOWU/EAST/pths/model_epoch_600.pth'
    
    # 输出标注文件目录
    output_dir = '/home/dengbinquan/XIAOWU/EAST/data/test_data'
    
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 调用处理函数
    process_two_images(img_path1, img_path2, model, device, output_dir)
