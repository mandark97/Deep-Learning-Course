import pandas as pd
import cv2
import numpy as np


def load_dataset():
    df = pd.read_csv("unibuc-2019-m2-cv/train_labels.txt")
    df['img'] = df.apply(lambda row: cv2.imread(
        form_img_path(row['id'])), axis=1)
    X = df['img'].values
    X = np.concatenate(X).reshape(-1, 224, 224, 3)
    y = df['class'].values

    return X, y


def form_img_path(x, path="unibuc-2019-m2-cv/data/data"):
    return f"{path}/{'0'* (6-len(str(x)))}{x}.png"
