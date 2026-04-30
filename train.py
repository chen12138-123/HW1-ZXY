import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import argparse
import random
from dataset import scan_eurosat, split_dataset
from model import extract_features, create_classifier
import joblib

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def featurize(paths, image_size: int, color_bins: int, edge_bins: int, progress_every: int = 500):
    X = np.zeros((len(paths), 3 * color_bins + 256 + edge_bins), dtype=np.float32)
    for i, p in enumerate(paths):
        X[i] = extract_features(p, image_size=image_size, color_bins=color_bins, edge_bins=edge_bins)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"Featurizing: {i+1}/{len(paths)}")
    return X


def accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def plot_confusion_matrix(y_true, y_pred, classes, out_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(out_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="EuroSAT_RGB")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--color_bins", type=int, default=16)
    parser.add_argument("--edge_bins", type=int, default=16)
    parser.add_argument("--classifier", type=str, default="linear_svm", choices=["linear_svm", "logistic", "rf"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_out", type=str, default="best_model.joblib")
    parser.add_argument("--max_train", type=int, default=0)
    parser.add_argument("--max_val", type=int, default=0)
    parser.add_argument("--max_test", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    image_paths, labels, classes = scan_eurosat(args.data_dir)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_dataset(
        image_paths, labels, split_ratio=(0.8, 0.1, 0.1), seed=args.seed
    )

    if args.max_train and args.max_train < len(train_paths):
        idx = np.random.RandomState(args.seed).choice(len(train_paths), size=args.max_train, replace=False)
        train_paths = [train_paths[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]
    if args.max_val and args.max_val < len(val_paths):
        idx = np.random.RandomState(args.seed + 1).choice(len(val_paths), size=args.max_val, replace=False)
        val_paths = [val_paths[i] for i in idx]
        val_labels = [val_labels[i] for i in idx]
    if args.max_test and args.max_test < len(test_paths):
        idx = np.random.RandomState(args.seed + 2).choice(len(test_paths), size=args.max_test, replace=False)
        test_paths = [test_paths[i] for i in idx]
        test_labels = [test_labels[i] for i in idx]

    X_train = featurize(train_paths, args.image_size, args.color_bins, args.edge_bins)
    X_val = featurize(val_paths, args.image_size, args.color_bins, args.edge_bins, progress_every=0)
    X_test = featurize(test_paths, args.image_size, args.color_bins, args.edge_bins, progress_every=0)

    clf = create_classifier(args.classifier, seed=args.seed)
    clf.fit(X_train, train_labels)

    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    train_acc = accuracy(train_labels, train_pred)
    val_acc = accuracy(val_labels, val_pred)
    test_acc = accuracy(test_labels, test_pred)

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Val Accuracy:   {val_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")

    joblib.dump(
        {
            "model": clf,
            "classes": classes,
            "image_size": args.image_size,
            "color_bins": args.color_bins,
            "edge_bins": args.edge_bins,
            "classifier": args.classifier,
            "seed": args.seed,
        },
        args.model_out,
    )
    plot_confusion_matrix(test_labels, test_pred, classes, out_path="confusion_matrix.png")
