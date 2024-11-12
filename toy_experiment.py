import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from raster import SimplifiedRasterModel as RasterModel
from vector import EnhancedVectorModel as VectorModel
import os
from skimage.draw import bezier_curve
from generate_data import generate_data, draw_line

# 设置随机种子
np.random.seed(2024)
torch.manual_seed(2024)
epochs = 10  # 总的训练epoch数

# 创建用于保存图片的文件夹
os.makedirs('raster_correct_classifications', exist_ok=True)
os.makedirs('raster_incorrect_classifications', exist_ok=True)
os.makedirs('vector_correct_classifications', exist_ok=True)
os.makedirs('vector_incorrect_classifications', exist_ok=True)


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, images, vectors, labels, mode='raster'):
        self.mode = mode
        self.labels = torch.tensor(labels, dtype=torch.long)
        if mode == 'raster':
            self.data = torch.tensor(images, dtype=torch.float32).unsqueeze(1)  # [N, 1, 64, 64]
        elif mode == 'vector':
            self.data = torch.tensor(vectors, dtype=torch.float32)  # [N, 64]
        else:
            raise ValueError("Mode should be 'raster' or 'vector'")
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

from thop import profile

# 定义输入张量
input_img = torch.randn(1, 1, 64, 64)
input_vec = torch.randn(1, 8)

# 实例化模型
raster_model = RasterModel()
vector_model = VectorModel()

# 计算 RasterModel 的 FLOPs 和参数
flops_raster, params_raster = profile(raster_model, inputs=(input_img,))
print(f"RasterModel - FLOPs: {flops_raster}, Params: {params_raster}")

# 计算 VectorModel 的 FLOPs 和参数
flops_vector, params_vector = profile(vector_model, inputs=(input_vec,))
print(f"VectorModel - FLOPs: {flops_vector}, Params: {params_vector}")

# 数据集大小
train_sizes = [100, 1000, 10000, 100000]
overall_results = {'raster': [], 'vector': []}

for size in train_sizes:
    # 生成数据
    total_samples = size + 2000  # 留出 2000 个用于测试
    images, vectors, labels = generate_data(total_samples)
    images = images / 255.0  # 归一化
    vectors = vectors / 64.0  # 归一化（假设坐标在 0-64 之间）
    
    # 划分训练集和测试集
    X_img_train, X_img_test, X_vec_train, X_vec_test, y_train, y_test = train_test_split(
        images, vectors, labels, test_size=2000, random_state=42)
    
    # 取所需的训练集大小
    X_img_train = X_img_train[:size]
    X_vec_train = X_vec_train[:size]
    y_train = y_train[:size]
    
    # 创建数据集和数据加载器
    batch_size = 64  # 可根据需要调整
    raster_train_dataset = CustomDataset(X_img_train, None, y_train, mode='raster')
    raster_test_dataset = CustomDataset(X_img_test, None, y_test, mode='raster')
    raster_train_loader = DataLoader(raster_train_dataset, batch_size=batch_size, shuffle=True)
    raster_test_loader = DataLoader(raster_test_dataset, batch_size=batch_size)
    
    vector_train_dataset = CustomDataset(None, X_vec_train, y_train, mode='vector')
    vector_test_dataset = CustomDataset(None, X_vec_test, y_test, mode='vector')
    vector_train_loader = DataLoader(vector_train_dataset, batch_size=batch_size, shuffle=True)
    vector_test_loader = DataLoader(vector_test_dataset, batch_size=batch_size)
    
    # ==================== Raster Model ====================
    raster_model = RasterModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(raster_model.parameters(), lr=0.001)
    
    # 存储每个 epoch 的准确率
    raster_accuracies = []
    
    # 训练 Raster 模型
    for epoch in range(1, epochs + 1):
        raster_model.train()
        for inputs, targets in raster_train_loader:
            optimizer.zero_grad()
            outputs = raster_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 每个 epoch 评估一次
        raster_model.eval()
        correct = 0
        total = 0

        # 新增：计数器，用于限制保存的案例数量
        correct_saved = 0
        incorrect_saved = 0
        max_cases_to_save = 5  # 每种类型最多保存5个案例

        with torch.no_grad():
            for inputs, targets in raster_test_loader:
                outputs = raster_model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                # 遍历当前批次的数据，保存案例
                for i in range(inputs.size(0)):
                    if preds[i] == targets[i]:
                        # 分类正确
                        if correct_saved < max_cases_to_save:
                            img = inputs[i].squeeze().cpu().numpy() * 255  # 恢复像素值
                            img = img.astype(np.uint8)
                            plt.imsave(f'raster_correct_classifications/size_{size}_epoch_{epoch}_idx_{i}.png', img, cmap='gray')
                            correct_saved += 1
                    else:
                        # 分类错误
                        if incorrect_saved < max_cases_to_save:
                            img = inputs[i].squeeze().cpu().numpy() * 255  # 恢复像素值
                            img = img.astype(np.uint8)
                            plt.imsave(f'raster_incorrect_classifications/size_{size}_epoch_{epoch}_idx_{i}.png', img, cmap='gray')
                            incorrect_saved += 1

                    # 如果达到保存数量限制，退出循环
                    if correct_saved >= max_cases_to_save and incorrect_saved >= max_cases_to_save:
                        break

                # 如果达到保存数量限制，退出循环
                if correct_saved >= max_cases_to_save and incorrect_saved >= max_cases_to_save:
                    break

        acc = correct / total
        raster_accuracies.append(acc)
        
        print(f"Epoch {epoch} - Raster model accuracy: {acc:.4f}")
    
    # 将结果存储在 overall_results 中
    overall_results['raster'].append(raster_accuracies)
    
    print('-' * 30)
    
    # ==================== Vector Model ====================
    vector_model = VectorModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vector_model.parameters(), lr=0.001)
    
    # 存储每个 epoch 的准确率
    vector_accuracies = []
    
    # 训练 Vector 模型
    for epoch in range(1, epochs + 1):
        vector_model.train()
        for inputs, targets in vector_train_loader:
            optimizer.zero_grad()
            outputs = vector_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 每个 epoch 评估一次
        vector_model.eval()
        correct = 0
        total = 0

        # 新增：计数器，用于限制保存的案例数量
        correct_saved = 0
        incorrect_saved = 0
        max_cases_to_save = 5  # 每种类型最多保存5个案例

        with torch.no_grad():
            for inputs, targets in vector_test_loader:
                outputs = vector_model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

                # 遍历当前批次的数据，保存案例
                for i in range(inputs.size(0)):
                    vector_data = inputs[i].cpu().numpy() * 64  # 恢复原始坐标值
                    flat_curve_points = vector_data.astype(int)

                    # 重建图像用于可视化
                    img = np.zeros((64, 64), dtype=np.uint8)

                    # 绘制曲线
                    curve_points = list(zip(flat_curve_points[::2], flat_curve_points[1::2]))
                    for x, y in curve_points:
                        if 0 <= x < 64 and 0 <= y < 64:
                            img[y, x] = 255

                    if preds[i] == targets[i]:
                        # 分类正确
                        if correct_saved < max_cases_to_save:
                            plt.imsave(f'vector_correct_classifications/size_{size}_epoch_{epoch}_idx_{i}.png', img, cmap='gray')
                            correct_saved += 1
                    else:
                        # 分类错误
                        if incorrect_saved < max_cases_to_save:
                            plt.imsave(f'vector_incorrect_classifications/size_{size}_epoch_{epoch}_idx_{i}.png', img, cmap='gray')
                            incorrect_saved += 1

                    # 如果达到保存数量限制，退出循环
                    if correct_saved >= max_cases_to_save and incorrect_saved >= max_cases_to_save:
                        break

                # 如果达到保存数量限制，退出循环
                if correct_saved >= max_cases_to_save and incorrect_saved >= max_cases_to_save:
                    break

        acc = correct / total
        vector_accuracies.append(acc)
        
        print(f"Epoch {epoch} - Vector model accuracy: {acc:.4f}")
    
    # 将结果存储在 overall_results 中
    overall_results['vector'].append(vector_accuracies)
    
    print('-' * 30)
    
    # 绘制结果
    epochs_range = range(1, epochs + 1)
    
    # 为当前训练集大小创建单独的图像
    plt.figure(figsize=(12, 6))
    
    # 绘制 Raster 模型的结果
    plt.plot(epochs_range, overall_results['raster'][-1], marker='o', label='Raster Model')
    
    # 绘制 Vector 模型的结果
    plt.plot(epochs_range, overall_results['vector'][-1], marker='s', label='Vector Model')
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Model Performance over Epochs (Train Size {size})')
    plt.legend()
    plt.grid(True)
    
    # 保存图像到文件，文件名包含训练集大小
    plt.savefig(f'model_performance_{size}_{epochs}_epochs.png')
    
    # 显示图像
    plt.show()
