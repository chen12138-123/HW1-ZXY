import os
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.image_paths.append(os.path.join(cls_dir, img_name))
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(root_dir, batch_size=64, split_ratio=(0.8, 0.1, 0.1)):
    # 基础预处理
    base_transform = [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    
    # 训练集增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        *base_transform
    ])
    
    val_test_transform = transforms.Compose(base_transform)
    
    # 先加载不带 transform 的数据集，手动应用不同的 transform
    full_dataset = EuroSATDataset(root_dir)
    train_size = int(split_ratio[0] * len(full_dataset))
    val_size = int(split_ratio[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # 包装 Subset 以应用不同的 transforms
    class ApplyTransform(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_loader = DataLoader(ApplyTransform(train_subset, train_transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ApplyTransform(val_subset, val_test_transform), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ApplyTransform(test_subset, val_test_transform), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, full_dataset.classes
