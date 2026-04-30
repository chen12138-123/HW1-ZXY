import matplotlib.pyplot as plt
import os
import numpy as np
import joblib
from dataset import scan_eurosat, split_dataset
from model import extract_features, load_rgb_image

def save_misclassified_samples(model_path, root_dir, output_path='error_examples/misclassified_samples.png', seed: int = 42):
    payload = joblib.load(model_path)
    model = payload["model"]
    classes = payload["classes"]
    image_size = int(payload.get("image_size", 64))
    color_bins = int(payload.get("color_bins", 16))
    edge_bins = int(payload.get("edge_bins", 16))
    feature_set = payload.get("feature_set", "basic")
    hog_cell = int(payload.get("hog_cell", 8))
    hog_block = int(payload.get("hog_block", 2))
    hog_bins = int(payload.get("hog_bins", 9))

    image_paths, labels, _ = scan_eurosat(root_dir)
    _, _, (test_paths, test_labels) = split_dataset(image_paths, labels, split_ratio=(0.8, 0.1, 0.1), seed=seed)

    os.makedirs('error_examples', exist_ok=True)
    
    misclassified = []
    for p, y in zip(test_paths, test_labels):
        x = extract_features(
            p,
            image_size=image_size,
            color_bins=color_bins,
            edge_bins=edge_bins,
            feature_set=feature_set,
            hog_cell=hog_cell,
            hog_block=hog_block,
            hog_bins=hog_bins,
        ).reshape(1, -1)
        pred = int(model.predict(x)[0])
        if pred != int(y):
            img = load_rgb_image(p, image_size=image_size)
            misclassified.append((img, classes[int(y)], classes[pred]))
        if len(misclassified) >= 6:
            break

    # 可视化
    plt.figure(figsize=(15, 10))
    for i, (img, true_label, pred_label) in enumerate(misclassified):
        plt.subplot(2, 3, i+1)
        plt.imshow(np.clip(img, 0, 1))
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')
    
    plt.suptitle('Misclassified Samples')
    plt.savefig(output_path)
    plt.close()
    print(f"Saved misclassified samples to {output_path}")

if __name__ == "__main__":
    save_misclassified_samples('best_model.joblib', 'EuroSAT_RGB')
