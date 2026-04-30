import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from model import create_model
from dataset import get_dataloaders
import os
import argparse
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean().item()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=150, amp=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_correct / len(train_loader.dataset)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_acc_sum = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_acc_sum += accuracy_top1(outputs, labels) * inputs.size(0)
                
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc = val_acc_sum / len(val_loader.dataset)
        
        if scheduler:
            scheduler.step()
            
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(float(epoch_acc))
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(float(val_epoch_acc))
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Acc: {epoch_acc:.4f} | Val Acc: {val_epoch_acc:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), 'best_model.pth')

    return history

def evaluate_model(model, test_loader, classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f'Final Test Accuracy: {test_acc:.4f}')
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png'); plt.close()

def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train'); plt.plot(history['val_loss'], label='Val')
    plt.title('Loss'); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train'); plt.plot(history['val_acc'], label='Val')
    plt.title('Accuracy'); plt.legend()
    plt.savefig('learning_curves.png'); plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mlp"])
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    num_classes = 10
    normalize = "imagenet" if args.model != "mlp" else "half"
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size if args.model != "mlp" else 64,
        normalize=normalize,
        num_workers=args.num_workers,
    )

    if args.model == "mlp":
        input_size = 64 * 64 * 3
        hidden_dims = [4096, 2048, 1024, 512, 256]
        model = create_model(
            "mlp",
            num_classes=num_classes,
            pretrained=False,
            mlp_input_size=input_size,
            mlp_hidden_dims=hidden_dims,
            mlp_dropout=0.2,
        )
    else:
        model = create_model(args.model, num_classes=num_classes, pretrained=bool(args.pretrained))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=max(args.lr * 0.01, 1e-6))
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=args.epochs, amp=not args.no_amp)
    plot_history(history)
    evaluate_model(model, test_loader, classes)
