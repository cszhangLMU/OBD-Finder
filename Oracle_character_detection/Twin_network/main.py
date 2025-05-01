import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

# 设置环境变量
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用GPU卡1
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 调试CUDA错误

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据集类
class OracleDataset(Dataset):
    def __init__(self, img_dir, annotation_files, transform=None):
        self.img_dir = img_dir
        self.annotation_files = annotation_files
        self.transform = transform
        self.data = self.load_annotations()

    def load_annotations(self):
        data = []
        for annotation_file in self.annotation_files:
            tree = ET.parse(annotation_file)
            root = tree.getroot()
            image_file = root.find('filename').text
            image_path = os.path.join(self.img_dir, image_file)
            boxes = []

            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                boxes.append((xmin, ymin, xmax, ymax))

            data.append({'image_path': image_path, 'boxes': boxes})

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]['image_path']
        image = Image.open(img_path).convert('RGB')
        boxes = self.data[idx]['boxes']

        cropped_images = []
        for box in boxes:
            cropped_image = image.crop(box)
            if self.transform:
                cropped_image = self.transform(cropped_image)
            cropped_images.append(cropped_image)

        return cropped_images, boxes

# 符号对生成
def create_pairs(dataset):
    pairs = []
    labels = []
    
    for i in range(len(dataset)):
        img1_crops, boxes1 = dataset[i]
        for j in range(i, len(dataset)):
            img2_crops, boxes2 = dataset[j]
            
            for crop1, box1 in zip(img1_crops, boxes1):
                for crop2, box2 in zip(img2_crops, boxes2):
                    pairs.append((crop1, crop2, box1, box2))
                    if i == j:
                        labels.append((1, 1, 1))  # 正样本
                    else:
                        labels.append((0, 0, 0))  # 负样本

    return pairs, labels

# 符号对数据集
class OraclePairDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1, img2, box1, box2 = self.pairs[idx]
        label = self.labels[idx]

        if self.transform:
            img1 = self.transform(img1) if not isinstance(img1, torch.Tensor) else img1
            img2 = self.transform(img2) if not isinstance(img2, torch.Tensor) else img2

        # 转换bounding box
        img1_size = img1.size(1), img1.size(2)  # 获取 (width, height)
        box1 = self.convert_bbox(box1, img1_size)
        box2 = self.convert_bbox(box2, img1_size)

        return img1, img2, torch.tensor(box1, dtype=torch.float32), torch.tensor(box2, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def convert_bbox(self, box, image_size):
        xmin, ymin, xmax, ymax = box
        width, height = image_size
        cx = (xmin + xmax) / 2.0 / width
        cy = (ymin + ymax) / 2.0 / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        return [cx, cy, w, h]

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return x * out

# 高级孪生网络架构
class AdvancedSiameseNetwork(nn.Module):
    def __init__(self):
        super(AdvancedSiameseNetwork, self).__init__()
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        
        self.se_block = SEBlock(2048)

        self.additional_conv = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 计算展平后的特征向量尺寸
        self._initialize_fc_layers()

    def _initialize_fc_layers(self):
        # 用于计算展平后特征向量的尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)  # 生成一个输入张量
            dummy_output = self.feature_extractor(dummy_input)
            dummy_output = self.se_block(dummy_output)
            dummy_output = self.additional_conv(dummy_output)
            flattened_size = dummy_output.view(-1).size(0)  # 展平后的大小
        
        # 使用计算出的 flattened_size 作为全连接层的输入大小
        self.fc_symbol = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_distribution = nn.Sequential(
            nn.Linear(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.fc_context = nn.Sequential(
            nn.Linear(flattened_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.se_block(x)
        x = self.additional_conv(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x1, x2, box1, box2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        
        # 符号相似性
        symbol_similarity = torch.abs(out1 - out2)
        symbol_similarity = self.fc_symbol(symbol_similarity)
        
        # 分布相似性
        distribution_input = torch.cat([box1, box2], dim=1)
        distribution_similarity = self.fc_distribution(distribution_input)
        
        # 上下文一致性
        context_similarity = torch.abs(out1 - out2)
        context_similarity = self.fc_context(context_similarity)
        
        return symbol_similarity, distribution_similarity, context_similarity

# 复杂的自定义损失函数
class AdvancedMultiTaskLoss(nn.Module):
    def __init__(self):
        super(AdvancedMultiTaskLoss, self).__init__()
        self.loss_weight_symbol = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.loss_weight_distribution = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.loss_weight_context = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, symbol_similarity, distribution_similarity, context_similarity, 
                target_symbol, target_distribution, target_context):
        loss_symbol = F.mse_loss(symbol_similarity, target_symbol)
        loss_distribution = F.mse_loss(distribution_similarity, target_distribution)
        loss_context = F.mse_loss(context_similarity, target_context)

        total_loss = (self.loss_weight_symbol * loss_symbol +
                      self.loss_weight_distribution * loss_distribution +
                      self.loss_weight_context * loss_context)
        return total_loss

# 高级训练与评估策略
from torch.optim.lr_scheduler import StepLR

def train_model(model, criterion, optimizer, scheduler, dataloader, num_epochs=50):
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (img1, img2, box1, box2, labels) in enumerate(dataloader):
            img1, img2 = img1.to(device), img2.to(device)
            box1, box2 = box1.to(device), box2.to(device)
            target_symbol, target_distribution, target_context = labels[:, 0], labels[:, 1], labels[:, 2]
            target_symbol = target_symbol.to(device).unsqueeze(1)
            target_distribution = target_distribution.to(device).unsqueeze(1)
            target_context = target_context.to(device).unsqueeze(1)

            optimizer.zero_grad()
            symbol_similarity, distribution_similarity, context_similarity = model(img1, img2, box1, box2)
            
            loss = criterion(symbol_similarity, distribution_similarity, context_similarity, 
                             target_symbol, target_distribution, target_context)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 打印进度
            if batch_idx % 10 == 0:  # 每10个batch打印一次
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

        avg_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Average Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')

        scheduler.step()

    print('Training completed.')



def evaluate_model(model, dataloader):
    model.eval()
    total_correct_symbol = 0
    total_correct_distribution = 0
    total_correct_context = 0
    total_samples = 0

    with torch.no_grad():
        for img1, img2, box1, box2, labels in dataloader:
            img1, img2 = img1.to(device), img2.to(device)
            box1, box2 = box1.to(device), box2.to(device)
            target_symbol, target_distribution, target_context = labels[:, 0], labels[:, 1], labels[:, 2]

            symbol_similarity, distribution_similarity, context_similarity = model(img1, img2, box1, box2)

            # 基于相似度的阈值判断是否相似
            pred_symbol = (symbol_similarity > 0.5).float()
            pred_distribution = (distribution_similarity > 0.5).float()
            pred_context = (context_similarity > 0.5).float()

            total_correct_symbol += (pred_symbol == target_symbol).sum().item()
            total_correct_distribution += (pred_distribution == target_distribution).sum().item()
            total_correct_context += (pred_context == target_context).sum().item()
            total_samples += pred_symbol.size(0)

    symbol_accuracy = total_correct_symbol / total_samples
    distribution_accuracy = total_correct_distribution / total_samples
    context_accuracy = total_correct_context / total_samples
    print(f'Symbol Accuracy: {symbol_accuracy * 100:.2f}%')
    print(f'Distribution Accuracy: {distribution_accuracy * 100:.2f}%')
    print(f'Context Accuracy: {context_accuracy * 100:.2f}%')

# 配置超参数
learning_rate = 0.001
num_epochs_per_batch = 10  # 每批训练的轮数
batch_size = 8
step_size = 10
gamma = 0.1

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 获取所有XML文件路径
annotation_dir = "/home/dengbinquan/XIAOWU/EAST/data/tags"
annotation_files = [os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith('.xml')]
print("Finished loading annotations.")

# 将文件列表分成5组，每组100个文件
num_batches = 5
batch_size_files = len(annotation_files) // num_batches
annotation_batches = [annotation_files[i:i + batch_size_files] for i in range(0, len(annotation_files), batch_size_files)]

# 模型、损失函数、优化器和调度器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AdvancedSiameseNetwork().to(device)
criterion = AdvancedMultiTaskLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# 逐批训练
for batch_idx, batch_annotation_files in enumerate(annotation_batches):
    print(f"Training batch {batch_idx + 1}/{num_batches} with {len(batch_annotation_files)} images...")
    
    # 加载数据
    train_dataset = OracleDataset(img_dir="/home/dengbinquan/XIAOWU/EAST/data/imgs", 
                                  annotation_files=batch_annotation_files, 
                                  transform=transform)
    pairs, labels = create_pairs(train_dataset)
    pair_dataset = OraclePairDataset(pairs, labels, transform=transform)
    train_loader = DataLoader(pair_dataset, batch_size=batch_size, shuffle=True)

    # 训练模型
    train_model(model, criterion, optimizer, scheduler, train_loader, num_epochs=num_epochs_per_batch)
    
    # 保存当前模型状态
    torch.save(model.state_dict(), f'model_batch_{batch_idx + 1}.pth')
    print(f"Batch {batch_idx + 1} completed and model saved.")

    # 评估当前模型
    evaluate_model(model, train_loader)

print("All batches completed.")
