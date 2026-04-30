import matplotlib.pyplot as plt
import numpy as np
from dataset import scan_eurosat
from model import load_rgb_image
import os

def visualize_class_mean_images(root_dir: str, image_size: int = 64, samples_per_class: int = 200, output_path: str = "class_means.png"):
    image_paths, labels, classes = scan_eurosat(root_dir)
    per_class = {i: [] for i in range(len(classes))}
    for p, y in zip(image_paths, labels):
        if len(per_class[y]) < samples_per_class:
            per_class[y].append(p)

    means = []
    for i in range(len(classes)):
        imgs = [load_rgb_image(p, image_size=image_size) for p in per_class[i]]
        mean_img = np.mean(np.stack(imgs, axis=0), axis=0) if imgs else np.zeros((image_size, image_size, 3), dtype=np.float32)
        means.append(mean_img)

    cols = 5
    rows = int(np.ceil(len(classes) / cols))
    plt.figure(figsize=(3 * cols, 3 * rows))
    for i, (cls, mean_img) in enumerate(zip(classes, means)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(np.clip(mean_img, 0, 1))
        plt.title(cls)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved class mean images to {output_path}")

if __name__ == "__main__":
    visualize_class_mean_images("EuroSAT_RGB", image_size=64)
