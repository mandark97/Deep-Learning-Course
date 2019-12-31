from tensorflow.keras.models import load_model
import tensorflow as tf
from utils import evaluate_model
from models import f1
from tensorflow.keras.optimizers import Adam
import pandas as pd


def run_from_save(model_name):
    dependencies = {
        'f1': f1
    }
    model_name = "densenet121_v2_fold0"
    model = load_model(f"models/{model_name}",
                       custom_objects=dependencies, compile=False)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', f1])
    evaluate_model(model, f"model_predictions/{model_name}")


def evaluate_folds(model_name):
    folds = []
    for fold in range(3):
        folds[fold] = pd.read_csv(
            f"model_predictions/{model_name}_fold{fold}.csv")
    df = pd.concat(folds)
    result_df = df.groupby('id').agg(lambda x: x.value_counts().index[0])
    result_df.to_csv(f"{model_name}.csv", index=False)
