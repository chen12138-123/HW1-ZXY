import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import MLP

def visualize_first_layer_weights(model_path, input_size, hidden_dims, num_classes, output_path='weight_visualization.png'):
    model = MLP(input_size, hidden_dims, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # 提取第一层权重
    weights = model.network[0].weight.data.numpy()
    
    num_neurons = 64
    plt.figure(figsize=(12, 12))
    for i in range(num_neurons):
        w = weights[i].reshape(3, 64, 64)
        w_min, w_max = w.min(), w.max()
        w = (w - w_min) / (w_max - w_min)
        w = np.transpose(w, (1, 2, 0))
        
        plt.subplot(8, 8, i+1)
        plt.imshow(w)
        plt.axis('off')
    
    plt.suptitle('First Hidden Layer Weight Visualization (High-Accuracy Model)')
    plt.savefig(output_path)
    plt.close()
    print(f"Weights visualization saved to {output_path}")

if __name__ == "__main__":
    input_size = 64 * 64 * 3
    hidden_dims = [4096, 2048, 1024, 512, 256] 
    num_classes = 10
    visualize_first_layer_weights('best_model.pth', input_size, hidden_dims, num_classes)
