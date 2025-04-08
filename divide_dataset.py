# -*- coding: utf-8 -*-
import os
import random
import shutil
from tqdm import tqdm

def split_dataset(root_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例加起来必须为1"

    classes = os.listdir(root_dir)
    for cls in tqdm(classes, desc="分配中"):
        img_paths = [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))]
        random.shuffle(img_paths)

        n_total = len(img_paths)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        splits = {
            'train': img_paths[:n_train],
            'val': img_paths[n_train:n_train + n_val],
            'test': img_paths[n_train + n_val:]
        }

        for split, paths in splits.items():
            save_dir = os.path.join(output_dir, split, cls)
            os.makedirs(save_dir, exist_ok=True)
            for p in paths:
                shutil.copy(p, save_dir)

# 用法示例
split_dataset(
    root_dir='D:/桌面/dataset',
    output_dir='D:/桌面/data',
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
