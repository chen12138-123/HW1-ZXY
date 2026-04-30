import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from model import MLP
from dataset import EuroSATDataset
import numpy as np

def save_misclassified_samples(model_path, root_dir, output_path='error_examples/misclassified_samples.png'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化并加载模型
    input_size = 64 * 64 * 3
    hidden_dims = [4096, 2048, 1024, 512, 256]
    num_classes = 10
    model = MLP(input_size, hidden_dims, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 测试集变换
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载数据集
    dataset = EuroSATDataset(root_dir, transform=test_transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    classes = dataset.classes

    os.makedirs('error_examples', exist_ok=True)
    
    misclassified = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = output.argmax(dim=1)
            
            if pred != y:
                img = x.cpu().squeeze().permute(1, 2, 0).numpy()
                img = img * 0.5 + 0.5
                img = np.clip(img, 0, 1)
                misclassified.append((img, classes[y.item()], classes[pred.item()]))
            
            if len(misclassified) >= 6:
                break

    # 可视化
    plt.figure(figsize=(15, 10))
    for i, (img, true_label, pred_label) in enumerate(misclassified):
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    
    plt.suptitle('Misclassified Samples (High-Accuracy Model)')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved misclassified samples to {output_path}")

if __name__ == "__main__":
    save_misclassified_samples('best_model.pth', 'EuroSAT_RGB')
