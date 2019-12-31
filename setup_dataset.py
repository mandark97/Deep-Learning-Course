import os

import pandas as pd


def make_folders_for_labels(data_path, labels_path, file_extension="png"):
    train_labels = pd.read_csv(labels_path, dtype=str)
    for label_class in train_labels['class'].unique():
        os.makedirs(f"{data_path}/{label_class}", exist_ok=True)

    # import pdb; pdb.set_trace()
    # train_labels
    for _, label in train_labels.iterrows():
        os.rename(f"{data_path}/{label['id']}.{file_extension}",
                   f"{data_path}/{label['class']}/{label['id']}.{file_extension}")

make_folders_for_labels("data/data", "data/train_labels.txt")
