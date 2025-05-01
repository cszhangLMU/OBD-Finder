import os
import urllib.request

# 指定要下载的模型的 URL
model_url = 'https://huggingface.co/langboat/mengzi-bert-L6-H768/resolve/main/pytorch_model.bin'

# 指定要保存模型文件的目录
model_dir = 'E:\\Image Matching\\LightGlue-main'

# 确保目录存在，如果不存在，则创建目录
os.makedirs(model_dir, exist_ok=True)

# 下载模型
model_file = os.path.join(model_dir, 'pytorch_model.bin')
urllib.request.urlretrieve(model_url, model_file)
