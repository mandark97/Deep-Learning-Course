from comet_ml import Experiment
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset import *
from models import *
from sklearn.metrics import confusion_matrix
from utils import *
from visualization import *

experiment = Experiment(api_key="i9Sew6Jy0Z36IZaUfJuR0cxhT",
                        project_name="general", workspace="mandark")
NFOLDS = 3
batch_size = 32
epochs = 15
learning_rate = 0.0005
model_name = "resnet101_model_v3"
params = {
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "model_name": model_name,
    "output_bias": False
}
experiment.log_parameters(params)

metrics_arr = []
dataset = load_dataset()
for fold in range(3):
    print(f"START FOR FOLD {fold}")
    with experiment.train():
        callbacks = [
            ModelCheckpoint(
                f"models/{model_name}{fold}_checkpoint", monitor='val_auc', save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_auc', patience=3, verbose=1,
                          restore_best_weights=True, mode='max'),
        ]
        train_dataset = crossval_ds(
            dataset, n_folds=NFOLDS, val_fold_idx=fold, training=True)
        test_dataset = crossval_ds(
            dataset, n_folds=NFOLDS, val_fold_idx=fold, training=False)
        weight, initial_bias = class_weight(train_dataset)
        print(weight)

        model = resnet101_model(learning_rate=learning_rate)
        history = model.fit(train_dataset.batch(batch_size),
                            epochs=epochs, verbose=1, validation_data=test_dataset.batch(batch_size), callbacks=callbacks,  class_weight=weight)

        plt = plot_metrics(history)
        experiment.log_figure(
            figure=plt, figure_name=f"Metrics History, Fold {fold}")

    with experiment.test():
        print("EVALUATE")
        scores = model.evaluate(
            test_dataset.batch(batch_size))
        metrics = dict(zip(model.metrics_names, scores))
        print(metrics)
        experiment.log_metrics(metrics, prefix=f"fold{fold}")

        print("MAKE CONFUSION MATRIX")
        y_true = get_labels(test_dataset)
        y_pred = model.predict(test_dataset.batch(batch_size))
        y_pred = parse_predict(y_pred)

        conf_matrix = confusion_matrix(y_true, y_pred)
        print(conf_matrix)

        plt = plot_confusion_matrix(conf_matrix, ['0', '1'])
        experiment.log_figure(
            figure=plt, figure_name=f"Confusion Matrix, Fold {fold}")

    print("SAVE AND EVALUATE")
    model.save(f"models/{model_name}_fold{fold}")
    evaluate_model(model, f"model_predictions/{model_name}_fold{fold}")

experiment.end()
