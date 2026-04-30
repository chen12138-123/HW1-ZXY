import os
from sklearn.model_selection import train_test_split


def scan_eurosat(root_dir: str):
    classes = sorted(
        [
            d
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
    )
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    image_paths = []
    labels = []
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            image_paths.append(os.path.join(cls_dir, img_name))
            labels.append(class_to_idx[cls_name])
    return image_paths, labels, classes


def split_dataset(image_paths, labels, split_ratio=(0.8, 0.1, 0.1), seed: int = 42):
    if abs(sum(split_ratio) - 1.0) > 1e-9:
        raise ValueError("split_ratio must sum to 1.0")

    train_ratio, val_ratio, test_ratio = split_ratio
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=(1.0 - train_ratio),
        random_state=seed,
        stratify=labels,
    )
    val_share = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=(1.0 - val_share),
        random_state=seed,
        stratify=temp_labels,
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)
