import torch
import matplotlib.pyplot as plt
import numpy as np
from model import MLP

def visualize_first_layer_weights(model_path, output_path='weight_visualization.png'):
    # 初始化模型参数
    input_size = 64 * 64 * 3
    hidden_dims = [1024, 512, 256]
    num_classes = 10
    
    # 加载模型
    model = MLP(input_size, hidden_dims, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # 获取第一层权重: network[0] 是第一个 Linear 层
    weights = model.network[0].weight.data.numpy() # 形状: (512, 12288)
    
    # 选择前 16 个神经元进行可视化
    num_neurons = 16
    plt.figure(figsize=(12, 12))
    
    for i in range(num_neurons):
        # 提取单个神经元的权重并 reshape 回图像尺寸 (C, H, W)
        # 原始输入是展平的 RGB，所以是 (3, 64, 64)
        weight_img = weights[i].reshape(3, 64, 64)
        
        # 转换为 (H, W, C) 以便 matplotlib 显示
        weight_img = weight_img.transpose(1, 2, 0)
        
        # 归一化到 [0, 1] 范围
        w_min, w_max = weight_img.min(), weight_img.max()
        weight_img = (weight_img - w_min) / (w_max - w_min)
        
        plt.subplot(4, 4, i + 1)
        plt.imshow(weight_img)
        plt.axis('off')
        plt.title(f'Neuron {i+1}')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Weights visualization saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    visualize_first_layer_weights('best_model.pth')
