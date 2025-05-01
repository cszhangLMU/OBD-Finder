import csv


# 读取文本文件并转换为表格格式
def convert_text_to_csv(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 创建 CSV 文件并写入表头
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Image1', 'Image2', 'Similarity'])

        # 逐行解析文本内容并写入 CSV 文件
        for line in lines:
            line = line.strip()  # 去掉换行符和多余的空格
            if ': ' not in line:
                print(f"Skipping invalid line (no ': '): {line}")  # 打印未匹配的行
                continue
            try:
                # 分割字符串
                pair, similarity = line.split(': ')
                if ', ' not in pair:
                    print(f"Skipping invalid pair format: {pair}")  # 打印无效的 pair 格式
                    continue
                image1, image2 = pair.split(', ')

                # 检查 similarity 是否为有效数值
                try:
                    similarity = float(similarity)
                except ValueError:
                    print(f"Skipping invalid similarity value: {similarity}")
                    continue

                csvwriter.writerow([image1, image2, similarity])
            except ValueError:
                print(f"Error processing line: {line}")  # 打印处理失败的行
                continue


# 使用示例
input_file = "C:\\Users\\XIAOWU\\Desktop\\result\\result_6.txt" # 输入的文本文件
output_file = "E:\\Edge detection\\Info\\total_results1\\results\\6\\data1\\output.csv"  # 输出的 CSV 文件
convert_text_to_csv(input_file, output_file)

print("转换完成，数据已保存到 output.csv 文件中。")
