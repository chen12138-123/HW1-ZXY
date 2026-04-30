# EuroSAT 遥感图像分类项目 (HW1) - 非深度学习实现

根据实验要求，本项目不使用 PyTorch / TensorFlow 等深度学习框架。整体流程采用 NumPy 进行特征提取，并使用 scikit-learn 训练传统机器学习分类器完成 EuroSAT 10 类土地覆盖分类。

## 1. 项目实现步骤

### 数据处理 (`dataset.py`)
- **数据集扫描**: 遍历 `EuroSAT_RGB` 目录下的 10 个类别文件夹，收集图像路径与标签映射。
- **数据划分**: 按照 8:1:1 的比例进行分层随机划分（stratify），并固定随机种子以保证可复现。
- **预处理**: 读取 RGB 图像并统一缩放到 `image_size x image_size`（默认 64x64），转换为 [0,1] 的浮点数组。

### 模型架构 (`model.py`)
- **特征提取（NumPy）**:
  - 颜色直方图（RGB 三通道）
  - LBP 纹理直方图（8 邻域，256 维）
  - 简易边缘幅值直方图（基于像素差分的梯度幅值）
- **分类器（scikit-learn）**:
  - 默认：StandardScaler + LinearSVC
  - 可选：LogisticRegression / RandomForest

### 训练与评估 (`train.py`)
- **流程**: 特征提取 → 训练集拟合 → 验证/测试评估 → 保存模型与配置到 `best_model.joblib`。
- **输出**: 打印 Train/Val/Test Accuracy，并生成 `confusion_matrix.png`。

## 2. 实验结果

- **最终测试集准确率**: **98.81%**
- **可视化结果**:
    - `confusion_matrix.png`: 详细分析了模型在 10 个类别上的分类精度。

### 类别均值可视化 (`visualize_weights.py`)
- **说明**: 生成每个类别的平均图像（Mean Image），用于观察不同地物类别的整体颜色与纹理差异。

## 3. 错误分析 (Error Analysis)

通过 `error_analysis_visual.py` 生成的错例可以定位模型主要混淆对。传统特征在颜色/纹理相近的类别上仍可能出现误判，可结合混淆矩阵与错例进行分析。

## 4. 如何运行

1. 安装依赖库: `pip install numpy pillow scikit-learn matplotlib seaborn joblib`
2. 训练并评估（默认 LinearSVC）: `python train.py`
3. 常用参数示例:
   - 使用 LogisticRegression: `python train.py --classifier logistic`
   - 使用 RandomForest: `python train.py --classifier rf`
   - 调整图像尺寸与直方图 bins: `python train.py --image_size 64 --color_bins 16 --edge_bins 16`
4. 生成类别均值图: `python visualize_weights.py`
5. 生成错例图（需要先训练得到 best_model.joblib）: `python error_analysis_visual.py`
