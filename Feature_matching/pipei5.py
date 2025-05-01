from pathlib import Path
import os
import torch
from lightglue import SuperPoint, LightGlue
from lightglue.utils import load_image, rbd
import pickle
import gzip
import cv2
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset

# 设置环境变量和设备
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载提取器和匹配器
extractor = SuperPoint(max_num_keypoints=2000).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# 图像文件夹路径
images_dir = Path("/data/23_24")
image_files = list(images_dir.glob("*.png"))
L = len(image_files)
print(f"找到 {len(image_files)} 张图像文件。")
#print("所有基准图像的索引：")
#for idx, img_path in enumerate(image_files):
#    print(f"索引 {idx}: {img_path.name}")

# output_image_order_file = Path("base_image_order.txt")
# with open(output_image_order_file, "w") as f:
#     f.write(f"找到 {len(image_files)} 张图像文件。\n")
#     f.write("基准图像顺序：\n")
#     for idx, img in enumerate(image_files):
#         line = f"{idx}: {img.name}\n"
#         print(line.strip())  # 打印到控制台
#         f.write(line)  # 写入文件
def save_progress(progress_file, start_i, completed_pairs):
    """保存进度文件"""
    with open(progress_file, "wb") as pf:
        progress_data = {'start_i': start_i, 'completed_pairs': completed_pairs}
        pickle.dump(progress_data, pf)
        # print(f"进度已保存：外层循环索引={start_i}, 已完成的图像对数={len(completed_pairs)}")
 

# 输出文件路径
output_file = Path("/result/r23.txt")
progress_file = Path("/process/r23.pkl")


# 创建一个缓存字典，用于存储每张图像的特征
features_cache = {}

def extract_and_cache_features(image_path, extractor, device):
    """ 提取图像特征并缓存 """
    if image_path in features_cache:
        return features_cache[image_path]  # 如果特征已经缓存，直接返回缓存
    image = load_image(image_path)  # 加载图像
    image = image.to(device)  # 确保图像也移动到正确的设备
    feats = extractor.extract(image)  # 提取图像特征
    features_cache[image_path] = feats  # 将特征存入缓存
    return feats



# 加载和保存进度的函数
def load_progress(progress_file):
    """加载进度文件"""
    if progress_file.exists():
        with open(progress_file, "rb") as pf:
            progress_data = pickle.load(pf)
            start_i = progress_data.get('start_i', 0)
            completed_pairs = progress_data.get('completed_pairs', set())
        return start_i, completed_pairs
    else:
        return 0, set()  # 初始状态


def save_matching_result(base_img, compare_img_path, matches, output_file):
    """保存匹配结果到文件"""
    if len(matches) > 0:  # 可加入匹配数目限制
        with open(output_file, "a") as f:
            f.write(f"{base_img.name}, {compare_img_path.name}, Matches: {len(matches)}\n")
        print(f"已保存匹配结果：{base_img.name} 与 {compare_img_path.name}")

# 恢复进度
if progress_file.exists():
    start_i, completed_pairs = load_progress(progress_file)
    print(f"恢复进度：start_i={start_i}, 已完成的图片对数量={len(completed_pairs)}")
else:
    start_i = 0  # 恢复的基准图像索引
    completed_pairs = set()
    

# 计算总的图像对数量，用于进度条
total_pairs = sum(1 for i in range(len(image_files)) for j in range(i + 1, len(image_files)))
print(f"总图像对数量: {total_pairs}")


# 预加载所有图像的特征
def load_all_features(image_files, extractor, device):
    all_features = {}
    for image_path in tqdm(image_files, desc="Loading and Extracting Features"):
        feats = extract_and_cache_features(image_path, extractor, device)  # 提取并缓存每张图像的特征
        all_features[image_path] = feats  # 存储到字典中
    return all_features


# 提前加载所有图像特征
all_features = load_all_features(image_files, extractor, device)



# 定义外层数据集，用于逐一加载基准图片
class BaseImageDataset(Dataset):
    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        return load_image(image_path)

# 定义内层数据集，每次加载基准图像后面的图片
class PairDataset(Dataset):
    def __init__(self, base_image, compare_images):
        self.base_image = base_image
        self.compare_images = compare_images

    def __len__(self):
        return len(self.compare_images)

    def __getitem__(self, idx):
        compare_image_path = self.compare_images[idx]
        compare_image = load_image(compare_image_path)
        return self.base_image, compare_image, compare_image_path  # 返回路径以便标识

# 图像对齐函数
def compute_homography_and_warp(image1, image2, kpts1, kpts2, matches):
    if matches.size == 0:
        return None, None, None

    points1 = np.float32([kpts1[m[0]].cpu().numpy() for m in matches])
    points2 = np.float32([kpts2[m[1]].cpu().numpy() for m in matches])

    if len(points1) < 4 or len(points2) < 4:
        return None, None, None

    H, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    if H is None:
        return None, None, None

    height, width = image2.shape[:2]
    warped_image = cv2.warpPerspective(image1, H, (width, height))

    # 确保 warped_image 是三维的
    if warped_image.ndim == 2:
        warped_image = warped_image[:, :, np.newaxis]

    return warped_image, H, mask

# 构建外层 DataLoader（不使用 pin_memory）
base_dataset = BaseImageDataset(image_files)
base_dataloader = DataLoader(base_dataset, batch_size=1, shuffle=False)

# 打开文件进行写入操作
try:
    with open(output_file, "a") as f, tqdm(total=total_pairs, desc="Processing Image Pairs") as pbar:
        # 外层循环：基准图片的遍历
        for i, base_image_tensor in enumerate(base_dataloader):
            if i < start_i:
                pbar.update(len(image_files) - i - 1)  # 更新进度条
                continue  # 跳过已处理的图片

            base_image_path = image_files[i]
            if i < L:
                compare_images_path = image_files[i+1]
            
            base_image_tensor = base_image_tensor[0].to(device, non_blocking=True)  # 提取基准图片张量
            print(f"Processing base image {image_files[i].name}")

            # 内层循环：基准图片之后的图片对逐一匹配
            compare_images = image_files[i + 1:]
            pair_dataset = PairDataset(base_image_tensor, compare_images)
            pair_dataloader = DataLoader(pair_dataset, batch_size=3000, num_workers=0, collate_fn=lambda x: x)

            # 处理每对图片
            for batch_index, batch in enumerate(pair_dataloader):
                for base_image, compare_image, compare_image_path in batch:
                    compare_image = compare_image.to(device, non_blocking=True)
                    pair_id = (i, image_files.index(compare_image_path))  # 使用 compare_image_path

                    # 跳过已完成的图片对
                    if pair_id in completed_pairs:
                        pbar.update(1)
                        continue

                    # 特征提取和初步匹配
                    with autocast():
                        # feats0 = extractor.extract(base_image)
                        feats0 = all_features[base_image_path]
                        # feats1 = extractor.extract(compare_image)
                        feats1 = all_features[compare_image_path]
                        
                        matches01 = matcher({"image0": feats0, "image1": feats1})

                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
                    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]

                    # 计算匹配率
                    match_ratio_0 = len(matches) / len(kpts0) if len(kpts0) > 0 else 0
                    match_ratio_1 = len(matches) / len(kpts1) if len(kpts1) > 0 else 0
                    max_match_ratio = max(match_ratio_0, match_ratio_1)
                    #print(f"Match {image_files[i].name} and {compare_image_path.name} with  ratio {max_match_ratio:.2f}")
                    

                    # 匹配率符合要求时，进行图像对齐和二次匹配
                    if max_match_ratio > 0.20:
                        print(f"Match found between {image_files[i].name} and {compare_image_path.name} with max ratio {max_match_ratio:.2f}")

                        image0_np = base_image.permute(1, 2, 0).cpu().numpy()
                        image1_np = compare_image.permute(1, 2, 0).cpu().numpy()

                        # 比较特征点数量，决定对齐方向
                        if len(kpts0) <= len(kpts1):
                            # 将 image0 对齐到 image1
                            src_image = image0_np
                            dst_image = image1_np
                            src_kpts = kpts0
                            dst_kpts = kpts1
                            src_feats = feats0
                            dst_feats = feats1
                            src_image_tensor = base_image
                            dst_image_tensor = compare_image
                        else:
                            # 将 image1 对齐到 image0
                            src_image = image1_np
                            dst_image = image0_np
                            src_kpts = kpts1
                            dst_kpts = kpts0
                            src_feats = feats1
                            dst_feats = feats0
                            src_image_tensor = compare_image
                            dst_image_tensor = base_image

                        # 调整匹配对
                        adjusted_matches = matches.clone()
                        if len(kpts0) > len(kpts1):
                            # 交换匹配的索引
                            adjusted_matches = adjusted_matches[:, [1, 0]]

                        # 计算单应矩阵并对齐
                        warped_image, H, mask = compute_homography_and_warp(src_image, dst_image, src_kpts, dst_kpts, adjusted_matches)

                        if warped_image is not None:
                            # 确保 warped_image 是三维的
                            if warped_image.ndim == 2:
                                warped_image = warped_image[:, :, np.newaxis]

                            warped_image_tensor = torch.tensor(warped_image).permute(2, 0, 1).unsqueeze(0).float().to(device)
                            # 提取对齐后的特征
                            feats_warped = extractor.extract(warped_image_tensor)
                            # 对比的目标图像特征
                            feats_dst_new = extractor.extract(dst_image_tensor.unsqueeze(0))
                            # 进行匹配
                            matches_warped = matcher({"image0": feats_warped, "image1": feats_dst_new})

                            feats_warped, feats_dst_new, matches_warped = [rbd(x) for x in [feats_warped, feats_dst_new, matches_warped]]
                            match_rate_2 = len(matches_warped["matches"]) / len(feats_warped["keypoints"]) if len(feats_warped["keypoints"]) > 0 else 0

                            # 保存匹配结果
                            if match_rate_2 > max_match_ratio:
                                f.write(f"{image_files[i].name}, {compare_image_path.name}: {match_rate_2:.2f}\n")
                                f.flush()
                                print(f"Match found between {image_files[i].name} and {compare_image_path.name} with best ratio {match_rate_2:.2f}")

                    # 更新已完成的图片对
                    completed_pairs.add(pair_id)
                    pbar.update(1)

                    # 每处理100对图片对保存进度
                    if pbar.n % 100 == 0:
                        save_progress(progress_file, i, completed_pairs)

                # 清理未使用的 GPU 缓存
                torch.cuda.empty_cache()

except Exception as e:
    print(f"An error occurred: {e}")

# 最后保存一次进度
save_progress(progress_file, len(image_files), completed_pairs)
print(f"Matched pairs saved to {output_file}")
