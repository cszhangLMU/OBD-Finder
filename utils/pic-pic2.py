import cv2
import numpy as np
import pytesseract
from PIL import Image
import os


def detect_and_crop(image_path, output_dir):
    # 读取图像
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 自适应阈值
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 形态学操作（闭操作）以连接断开的部分
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 寻找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for contour in contours:
        # 过滤掉面积过小的轮廓
        area = cv2.contourArea(contour)
        if area < 500:  # 调整此值以排除过小的区域
            continue

        # 获取凸包
        hull = cv2.convexHull(contour)

        # 计算外接矩形
        x, y, w, h = cv2.boundingRect(hull)
        if w > 100 and h > 100:  # 过滤掉太小的区域
            count += 1
            cropped_image = image[y:y + h, x:x + w]

            # 识别下方的编号
            number_y_start = y + h
            number_y_end = number_y_start + 130  # 假设编号区域高度为130像素
            number_image = image[number_y_start:number_y_end, x:x + w]
            number_text = pytesseract.image_to_string(Image.fromarray(number_image), config='--psm 7').strip()

            # 如果识别的编号不是数字，使用默认编号
            if not number_text.isdigit():
                number_text = f"{os.path.splitext(os.path.basename(image_path))[0]}_{count}"

            # 保存裁剪后的图像
            output_path = os.path.join(output_dir, f"{number_text}.png")
            cropped_pil_image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            cropped_pil_image.save(output_path)


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(input_dir, filename)
            detect_and_crop(image_path, output_dir)

# 输入和输出目录
input_dir = "E:\\Edge detection\\Info\\26"
output_dir = "E:\\Edge detection\\Info\\total_results1\\26"

# 处理目录中的所有图像
process_directory(input_dir, output_dir)
