# import os
#
#
# def get_images(img_path):
#     '''
#     find image files in data path
#     :return: list of files found
#     '''
#     img_path = os.path.abspath(img_path)
#     files = []
#     exts = ['jpg', 'png', 'jpeg', 'JPG', 'PNG']
#     for parent, dirnames, filenames in os.walk(img_path):
#         for filename in filenames:
#             for ext in exts:
#                 if filename.endswith(ext):
#                     files.append(os.path.join(parent, filename))
#                     break
#     print('Find {} images'.format(len(files)))
#     return sorted(files)
#
#
# def get_txts(txt_path):
#     '''
#     find gt files in data path
#     :return: list of files found
#     '''
#     txt_path = os.path.abspath(txt_path)
#     files = []
#     exts = ['txt']
#     for parent, dirnames, filenames in os.walk(txt_path):
#         for filename in filenames:
#             for ext in exts:
#                 if filename.endswith(ext):
#                     files.append(os.path.join(parent, filename))
#                     break
#     print('Find {} txts'.format(len(files)))
#     return sorted(files)
#
#
# if __name__ == '__main__':
#     # Image and text file paths
#     img_path = "C:\\Users\\XIAOWU\\Desktop\\fsdownload\\test"
#     txt_path = "C:\\Users\\XIAOWU\\Desktop\\fsdownload\\test_data1"
#
#     files = get_images(img_path)
#     txts = get_txts(txt_path)
#
#     # Extract base filenames (without extensions)
#     img_files = set([os.path.splitext(os.path.basename(f))[0] for f in files])
#     txt_files = set([os.path.splitext(os.path.basename(f))[0] for f in txts])
#
#     # Find extra images without corresponding text files and vice versa
#     extra_images = img_files - txt_files  # Images with no matching text files
#     missing_txts = txt_files - img_files  # Text files with no matching images
#
#     # Display the mismatched files
#     if extra_images:
#         print("Extra images without txts:", extra_images)
#     if missing_txts:
#         print("Missing txts for images:", missing_txts)
#
#     # Check if the counts match, if not, proceed with matching pairs only
#     n = min(len(files), len(txts))
#     with open('test.txt', 'w') as f:
#         for i in range(n):
#             line = files[i] + '\t' + txts[i] + '\n'
#             f.write(line)
#
#     print('Dataset generated successfully ^_^ ')
import csv
import os

# 设置 CSV 文件路径
csv_file ="E:\\Edge detection\\Info\\total_results1\\results\\Yellow\\data4\\output.csv" # 假设 CSV 文件路径
output_file = "E:\\Edge detection\\Info\\total_results1\\results\\Yellow\\data4\\output1.csv"  # 输出处理后的文件列表路径

# 打开 CSV 文件并处理
with open(csv_file, mode='r') as file, open(output_file, mode='w', newline='') as output:
    reader = csv.reader(file)
    writer = csv.writer(output)

    # 逐行读取 CSV 内容
    for row in reader:
        new_row = []
        for filename in row:
            # 如果文件名中包含下划线，则替换为连字符并加上 -16
            if '_' in filename and filename.endswith('.png'):
                filename = filename.replace('_', '-')  # 替换 _ 为 -
                # filename = filename.replace('.png', '-17.png')  # 在 .png 后加 -16
            new_row.append(filename)

        # 将处理后的行写入输出文件
        writer.writerow(new_row)

print(f"文件已处理并保存到 {output_file}")
