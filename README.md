# EuroSAT 遥感图像分类项目 (HW1) - 最新训练版

本项目使用卷积神经网络 ResNet18 对 EuroSAT 数据集进行 10 类土地覆盖分类，并保留 MLP 作为对比基线。当前默认训练配置以提升精度与泛化为目标（更强的数据增强、ImageNet 归一化、AdamW + Cosine 学习率、Label Smoothing、AMP）。

## 1. 项目实现步骤

### 数据处理 (`dataset.py`)
- **数据集加载**: 自定义 `EuroSATDataset` 类。
- **数据划分**: 按照 8:1:1 的比例随机划分，并固定随机种子以保证可复现。
- **预处理（ResNet18 默认）**: 输入尺寸统一为 224x224，并采用 ImageNet 归一化（mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)）。
- **训练增强（ResNet18 默认）**: RandomResizedCrop、水平/垂直翻转、旋转、ColorJitter、RandomErasing；验证/测试仅 Resize + Normalize。
- **MLP 基线**: 若选择 `--model mlp`，输入尺寸为 64x64，并使用 0.5/0.5 的归一化方式。

### 模型架构 (`model.py`)
- **ResNet18（默认）**: 使用 torchvision 的 ResNet18，将最后分类层替换为 10 类输出；支持 `--pretrained 1` 使用 ImageNet 预训练权重初始化。
- **MLP（可选基线）**: 展平 64x64x3 输入并经过多层全连接 + BN + LeakyReLU + Dropout 的分类网络。

### 训练与评估 (`train.py`)
- **训练轮数**: 默认 150 个 Epoch（满足“训练 100 轮以上”的要求）。
- **优化器**: AdamW（默认 lr=3e-4，weight_decay=1e-4）。
- **学习率策略**: CosineAnnealingLR（余弦退火）。
- **损失函数**: CrossEntropyLoss(label_smoothing=0.1)。
- **AMP**: CUDA 环境默认开启混合精度训练，可用 `--no_amp` 关闭。
- **模型保存**: 以验证集准确率为准保存最优模型到 `best_model.pth`，并使用该权重进行测试集评估。

## 2. 实验结果

- **最终测试集准确率（当前 best_model.pth）**: **98.81%**（ResNet18，测试集评估结果）。
- **可视化结果**:
    - `learning_curves.png`: 展示了 Loss 和 Accuracy 的平稳优化过程。
    - `confusion_matrix.png`: 详细分析了模型在 10 个类别上的分类精度。

### 权重可视化与空间模式 (`visualize_weights.py`)
- **说明**: 该脚本用于可视化当前 `best_model.pth` 的第一层权重。若模型为 MLP，则可将权重恢复为 64x64x3 形式进行可视化；若模型为 ResNet18，可视化结果更适合解释卷积核对颜色/纹理的响应模式。

## 3. 错误分析 (Error Analysis)

通过 `error_analysis_visual.py` 生成的错例可以定位模型主要混淆对。即便在较高准确率下，某些类别仍可能因颜色/纹理相近、类别内部差异较大或空间结构相似而发生误判。

## 4. 如何运行

1. 确保安装了依赖库: `pip install torch torchvision matplotlib seaborn scikit-learn`
2. 运行训练脚本（默认 ResNet18 + 150 Epoch）: `python train.py`
3. 常用参数示例:
   - 使用预训练权重: `python train.py --model resnet18 --pretrained 1`
   - 指定轮数与 batch: `python train.py --epochs 150 --batch_size 64`
   - 显存不足时降分辨率: `python train.py --image_size 192`
   - 使用 MLP 基线: `python train.py --model mlp`
4. 运行可视化脚本: `python visualize_weights.py` 和 `python error_analysis_visual.py`
