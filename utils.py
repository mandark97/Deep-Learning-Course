import pandas as pd
import cv2
import numpy as np


def form_img_path(x, path="unibuc-2019-m2-cv/data/data"):
    return f"{path}/{'0'* (6-len(str(x)))}{x}.png"


def evaluate_model(model, model_name):
    img_list = [form_img_path(x) for x in range(17001, 22150)]
    X_test = np.array([cv2.imread(img) for img in img_list])
    y = model.predict(X_test, batch_size=32)
    ans_df = pd.DataFrame({'id': [f"{'0'* (6-len(str(x)))}{x}" for x in range(17001, 22150)],
                           'class': parse_predict(y)})
    ans_df.to_csv(f'{model_name}.csv', index=False)


def parse_predict(y_pred):
    return (y_pred.flatten() > 0.5).astype(int)
