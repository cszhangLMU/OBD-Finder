import os
import shutil
import csv
from PIL import Image
from siamese import Siamese
from pathlib import Path
from collections import defaultdict
# 设置环境变量和设备
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os
import shutil
import csv
from PIL import Image
from siamese import Siamese
from pathlib import Path
from collections import defaultdict

# 设置环境变量和设备
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def calculate_similarity(folder1_path, image_extension='.png'):
    """
    计算相似度并保存到每个子文件夹下，同时生成两个CSV文件：
    1. image_similarity_results.csv - 保存每个 Image_1 的最佳匹配结果。
    2. image_similarity_results1.csv - 保存所有匹配结果。

    根据平均相似度将子文件夹复制到不同的结果文件夹。
    
    :param folder1_path: 存放多个子文件夹的父文件夹路径。
    :param image_extension: 图像文件的扩展名（默认为 '.png'）。
    """
    # 创建存放结果的文件夹
    result_folders = {
        'A1': folder1_path / 'result_A1', 
        'B1': folder1_path / 'result_B1', 
        'C1': folder1_path / 'result_C1',
        'D1': folder1_path / 'result_D1',
    }

    # 确保 result_A1, result_B1, result_C1, result_D1 目录存在
    for folder in result_folders.values():
        if not folder.exists():
            folder.mkdir(parents=True)

    # 遍历 folder1_path 下的每个子文件夹 folder2
    for folder2 in folder1_path.iterdir():
        if folder2.is_dir():  # 确保是文件夹
            input_csv = folder2 / 'matched_boxes.csv'  # 输入的CSV文件路径
            images_folder = folder2 / 'splits'  # 图像文件夹路径
            output_csv = folder2 / 'image_similarity_results.csv'  # 输出最佳匹配结果的CSV文件路径
            output_csv1 = folder2 / 'image_similarity_results1.csv'  # 输出所有匹配结果的CSV文件路径

            if not input_csv.exists() or not images_folder.exists():
                print(f"Skipping folder {folder2} because required files are missing.")
                continue

            # 初始化Siamese模型
            model = Siamese()

            total_similarity = 0
            valid_matches_count = 0

            # 存储每个 Image_1 的所有匹配
            image_matches = defaultdict(list)

            # 打开输入的CSV文件读取图像文件名
            with open(input_csv, mode='r') as infile:
                reader = csv.reader(infile)
                header = next(reader, None)  # 跳过表头（如果有的话）

                # 读取每一对图像的文件名
                for row in reader:
                    if len(row) < 2:
                        print(f"Invalid row in {input_csv}: {row} (Skipping)")
                        continue

                    image_1_filename = row[0].strip() + image_extension  # 加上扩展名
                    image_2_filename = row[1].strip() + image_extension  # 加上扩展名

                    # 根据文件名拼接出图像的完整路径
                    image_1_path = images_folder / image_1_filename
                    image_2_path = images_folder / image_2_filename

                    try:
                        # 打开图片
                        image_1 = Image.open(image_1_path).convert('RGB')
                        image_2 = Image.open(image_2_path).convert('RGB')
                    except Exception as e:
                        print(f"Error opening images: {e} (Skipping pair {image_1_filename}, {image_2_filename})")
                        continue

                    # 计算图像相似度
                    probability = model.detect_image(image_1, image_2)

                    # 将每对匹配的相似度添加到字典中，以 Image_1 为 key
                    image_matches[image_1_filename].append((image_2_filename, probability.item()))

            # 打开输出的CSV文件保存所有相似度结果
            with open(output_csv1, mode='w', newline='') as outfile1:
                writer1 = csv.writer(outfile1)
                writer1.writerow(["Image_1", "Image_2", "Similarity"])

                # 对每个 Image_1 存储所有匹配结果
                for image_1_filename, matches in image_matches.items():
                    for match in matches:
                        image_2_filename, similarity = match
                        writer1.writerow([image_1_filename, image_2_filename, similarity])

                        total_similarity += similarity
                        valid_matches_count += 1

                        print(f"Processed: {image_1_filename} -> Match: {image_2_filename} -> Similarity: {similarity:.4f}")

            # 打开输出的CSV文件保存最佳匹配的相似度结果
            with open(output_csv, mode='w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["Image_1", "Image_2", "Similarity"])

                # 对每个 Image_1 选择最匹配的 Image_2
                for image_1_filename, matches in image_matches.items():
                    # 如果只有一个匹配对，直接使用
                    if len(matches) == 1:
                        best_match = matches[0]
                    else:
                        # 按相似度从高到低排序
                        matches_sorted = sorted(matches, key=lambda x: x[1], reverse=True)
                        # 选取相似度最高的匹配
                        best_match = matches_sorted[0]

                    image_2_filename, best_similarity = best_match

                    # 保存最佳匹配
                    writer.writerow([image_1_filename, image_2_filename, best_similarity])
                    total_similarity += best_similarity
                    valid_matches_count += 1

                    print(f"Processed Best Match: {image_1_filename} -> {image_2_filename} -> Similarity: {best_similarity:.4f}")

            # 计算平均相似度：对于所有最佳匹配，计算其平均值
            if valid_matches_count > 0:
                average_similarity = total_similarity / valid_matches_count
                print(f"Average similarity for {folder2.name}: {average_similarity:.4f}")
            else:
                average_similarity = 0
                print(f"No valid matches for {folder2.name}.")

            # 将平均相似度写入到 CSV 文件的最后一行
            with open(output_csv, mode='a', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(["Average Similarity", "", average_similarity])

            # 根据平均相似度决定是否复制到对应的结果文件夹
            if average_similarity >= 0.9:
                # 将子文件夹复制到 result_A1
                destination = result_folders['A1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_A1")
                shutil.copytree(folder2, destination)
            elif 0.8 <= average_similarity < 0.9:
                # 将子文件夹复制到 result_B1
                destination = result_folders['B1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_B1")
                shutil.copytree(folder2, destination)
            elif 0.7 <= average_similarity < 0.8:
                # 将子文件夹复制到 result_C1
                destination = result_folders['C1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_C1")
                shutil.copytree(folder2, destination)
            elif 0.6 <= average_similarity < 0.7:
                # 将子文件夹复制到 result_D1
                destination = result_folders['D1'] / folder2.name
                print(f"Average similarity {average_similarity:.4f} -> Copying {folder2.name} to result_D1")
                shutil.copytree(folder2, destination)
            else:
                # 平均相似度低于 0.6，继续原操作（不复制）
                print(f"Average similarity {average_similarity:.4f} -> Not copying {folder2.name}")

            print(f"Similarity results saved to {output_csv} and {output_csv1}\n")

if __name__ == "__main__":
    # 输入的文件夹路径，存放多个 folder2
    folder1_path = Path("/data/C1/all_folders30")  # 请替换为实际的文件夹路径

    # 调用计算函数
    calculate_similarity(folder1_path, image_extension='.png')  # 假设图像文件为 .png 格式
