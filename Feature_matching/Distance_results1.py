import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lightglue import SuperPoint, LightGlue
import os
import csv
# 设置环境变量和设备
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载SuperPoint提取器和LightGlue匹配器
extractor = SuperPoint(max_num_keypoints=2000).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# 图像加载函数
def load_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image

# 确保图像是灰度图像（单通道）
def ensure_single_channel(image):
    if image.ndim == 2:
        image = image[np.newaxis, np.newaxis, ...]  # (H, W) -> (1, 1, H, W)
    elif image.ndim == 3 and image.shape[0] == 1:
        image = image[np.newaxis, ...]  # (1, H, W) -> (1, 1, H, W)
    return image

# 计算单应性矩阵并进行透视变换
def compute_homography_and_warp(image1, image2, kpts1, kpts2, matches):
    if matches.size == 0:
        print("No matches found.")
        return None, None, None

    points1 = np.float32([kpts1[m[0]] for m in matches])
    points2 = np.float32([kpts2[m[1]] for m in matches])

    if len(points1) < 4 or len(points2) < 4:
        print(f"Insufficient points for homography: {len(points1)} points found.")
        return None, None, None

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)

    if H is None:
        print("Homography computation failed.")
        return None, None, None

    height, width = image2.shape[:2]
    warped_image = cv2.warpPerspective(image1, H, (width, height))

    return warped_image, H, mask

# 从txt文件中读取标注框 (格式为 x1,y1,x2,y2,x3,y3,x4,y4)
def load_polygon_annotations(file_path):
    polygons = []
    with open(file_path, 'r') as f:
        for line in f:
            coords = list(map(int, line.strip().split(',')))
            polygon = np.array([[coords[i], coords[i + 1]] for i in range(0, len(coords), 2)])
            polygons.append(polygon)
    return polygons

# 转换坐标到对齐后的图像中
def transform_points(points, H):
    num_points = points.shape[0]
    ones = np.ones((num_points, 1))
    points_ones = np.hstack([points, ones])
    transformed_points = H.dot(points_ones.T).T
    transformed_points /= transformed_points[:, 2][:, np.newaxis]
    return transformed_points[:, :2]

# 计算多边形的质心
def compute_centroid(polygon):
    polygon = np.array(polygon)
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    centroid_x = np.mean(x_coords)
    centroid_y = np.mean(y_coords)
    return np.array([centroid_x, centroid_y])

# 计算两个质心之间的欧氏距离
def compute_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

# 找到匹配的标注框：通过质心距离判断匹配
def find_matching_boxes_by_position(polygons1, polygons2, transformed_polygons, threshold=50):
    matched_boxes = []

    for i, transformed_polygon in enumerate(transformed_polygons):
        transformed_centroid = compute_centroid(transformed_polygon)

        for j, polygon2 in enumerate(polygons2):
            centroid2 = compute_centroid(polygon2)

            distance = compute_distance(transformed_centroid, centroid2)

            if distance < threshold:
                matched_boxes.append((i, j, distance))
                print(f"Box {i} from Image 0 matches with Box {j} from Image 1 with distance: {distance}")

    return matched_boxes

# 保存可视化结果到图片
def save_visualization(folder, image0, image1, polygons0, polygons1, matched_boxes, transformed_annotations):
    fig, axs = plt.subplots(1, 2, figsize=(15, 8))

    # 显示原始图像0和原始标注框
    axs[0].imshow(image0, cmap='gray')
    axs[0].set_title('Original Image 0 with Original Annotations')
    for idx, polygon in enumerate(polygons0):
        polygon = np.int32([polygon])  # 确保是二维数组
        axs[0].add_patch(plt.Polygon(polygon[0], fill=None, edgecolor='red', linewidth=1))
        axs[0].text(polygon[0][0][0], polygon[0][0][1], f'{idx}', color='blue', fontsize=10, weight='bold')

    # 显示原始图像1和原始标注框
    axs[1].imshow(image1, cmap='gray')
    axs[1].set_title('Original Image 1 with Original Annotations')
    for idx, polygon in enumerate(polygons1):
        polygon = np.int32([polygon])  # 确保是二维数组
        axs[1].add_patch(plt.Polygon(polygon[0], fill=None, edgecolor='red', linewidth=1))
        axs[1].text(polygon[0][0][0], polygon[0][0][1], f'{idx}', color='blue', fontsize=10, weight='bold')

    # 高亮显示匹配的标注框
    for match in matched_boxes:
        i, j, distance = match

        # 高亮图像0中匹配的标注框
        matched_polygon0 = np.int32([polygons0[i]])  # 确保是二维数组
        axs[0].add_patch(plt.Polygon(matched_polygon0[0], fill=None, edgecolor='green', linewidth=2, linestyle='--'))

        # 高亮图像1中匹配的标注框
        matched_polygon1 = np.int32([polygons1[j]])  # 确保是二维数组
        axs[1].add_patch(plt.Polygon(matched_polygon1[0], fill=None, edgecolor='blue', linewidth=2, linestyle='--'))

    # 保存图片到文件夹
    output_file = folder / "visualization.png"
    plt.savefig(output_file)
    plt.close()
    print(f"Visualization saved to {output_file}")

# 保存每个标注框的实际裁剪图像
def save_split_images(image0, image1, polygons0, polygons1, splits_folder, images):
    # 获取图像的文件名（不含扩展名）
    image0_name = images[0].stem
    image1_name = images[1].stem

    # 保存Image0的标注框
    for i, polygon in enumerate(polygons0):
        # 计算多边形的边界框（xmin, ymin, xmax, ymax）
        x_min = int(np.min(polygon[:, 0]))
        x_max = int(np.max(polygon[:, 0]))
        y_min = int(np.min(polygon[:, 1]))
        y_max = int(np.max(polygon[:, 1]))

        # 裁剪图像区域
        cropped_image = image0[y_min:y_max, x_min:x_max]

        # 检查裁剪图像是否为空
        if cropped_image.size == 0:
            print(f"Warning: Cropped image for Image0 Box {i} is empty.")
            continue  # 跳过空图像

        # 保存裁剪后的图像，文件名为image0_{i}
        filename = f"{image0_name}_{i}.png"
        cv2.imwrite(str(splits_folder / filename), cropped_image)  # 保存图像

    # 保存Image1的标注框
    for j, polygon in enumerate(polygons1):
        # 计算多边形的边界框（xmin, ymin, xmax, ymax）
        x_min = int(np.min(polygon[:, 0]))
        x_max = int(np.max(polygon[:, 0]))
        y_min = int(np.min(polygon[:, 1]))
        y_max = int(np.max(polygon[:, 1]))

        # 裁剪图像区域
        cropped_image = image1[y_min:y_max, x_min:x_max]

        # 检查裁剪图像是否为空
        if cropped_image.size == 0:
            print(f"Warning: Cropped image for Image1 Box {j} is empty.")
            continue  # 跳过空图像

        # 保存裁剪后的图像，文件名为image1_{j}
        filename = f"{image1_name}_{j}.png"
        cv2.imwrite(str(splits_folder / filename), cropped_image)  # 保存图像

# 处理所有子文件夹
def process_all_folders(base_folder):
    for folder in base_folder.iterdir():
        if not folder.is_dir():
            continue

        print(f"Processing folder: {folder}")

        # 获取目标图片和标注文件
        images = sorted([f for f in folder.glob("*.png") if f.stem in folder.name.split('_')])
        annotations = sorted([f for f in folder.glob("*.txt") if f.stem in folder.name.split('_')])

        if len(images) != 2 or len(annotations) != 2:
            print(f"Skipping folder {folder}, expected 2 target images and 2 corresponding annotations.")
            continue

        # 加载图片
        image0 = load_image(images[0])
        image1 = load_image(images[1])

        # 加载标注框
        polygons0 = load_polygon_annotations(annotations[0])
        polygons1 = load_polygon_annotations(annotations[1])

        # 确保标注框少的图像始终为 image0
        if len(polygons0) > len(polygons1):
            image0, image1 = image1, image0
            polygons0, polygons1 = polygons1, polygons0
            annotations = annotations[::-1]
            images = images[::-1]

        image0_tensor = ensure_single_channel(torch.from_numpy(image0).float() / 255.0).to(device)
        image1_tensor = ensure_single_channel(torch.from_numpy(image1).float() / 255.0).to(device)

        # 提取特征并匹配
        feats0 = extractor.extract(image0_tensor)
        feats1 = extractor.extract(image1_tensor)
        matches01 = matcher({"image0": feats0, "image1": feats1})

        kpts0, kpts1 = feats0["keypoints"][0].cpu().numpy(), feats1["keypoints"][0].cpu().numpy()
        matches = np.array(matches01["matches"][0].cpu())

        # 计算单应性并对齐
        warped_image, H, mask = compute_homography_and_warp(image0, image1, kpts0, kpts1, matches)

        if H is None:
            print(f"Skipping folder {folder}, homography could not be computed.")
            continue

        # 转换标注框到对齐后的图像中
        transformed_annotations = [transform_points(polygon, H) for polygon in polygons0]

        # 匹配标注框
        matched_boxes = find_matching_boxes_by_position(polygons0, polygons1, transformed_annotations)

        # 保存切割标注框图像
        splits_folder = folder / "splits"
        splits_folder.mkdir(parents=True, exist_ok=True)
        save_split_images(image0, image1, polygons0, polygons1, splits_folder, images)

        # 保存匹配信息
        output_csv = folder / "matched_boxes.csv"
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Image0_Box_Index", "Image1_Box_Index"])
            for match in matched_boxes:
                i, j, _ = match
                image0_name = images[0].stem  # 去掉.png后缀
                image1_name = images[1].stem  # 去掉.png后缀
                writer.writerow([f"{image0_name}_{i}", f"{image1_name}_{j}"])

        print(f"Matched boxes saved to {output_csv}")

        # 保存可视化结果
        save_visualization(folder, image0, image1, polygons0, polygons1, matched_boxes, transformed_annotations)


# 设置主文件夹路径并调用函数处理所有子文件夹
base_folder = Path(r"/data/C1/all_folders30")  # 请替换为实际的文件夹路径
process_all_folders(base_folder)
