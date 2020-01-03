from comet_ml import Experiment
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold

from dataset import *
from models import *
from sklearn.metrics import confusion_matrix
from utils import *
from visualization import *

experiment = Experiment(api_key="i9Sew6Jy0Z36IZaUfJuR0cxhT",
                        project_name="general", workspace="mandark")
NFOLDS = 3
batch_size = 32
epochs = 5
learning_rate = 0.0005
model_name = "densenet121_v4"
params = {
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "model_name": model_name,
    "output_bias": False
}
experiment.log_parameters(params)

metrics_arr = []
X, y = load_dataset()
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True)
fold = 0
for train_index, test_index in skf.split(X, y):
    print(f"START FOR FOLD {fold}")
    with experiment.train():
        callbacks = [
            # ModelCheckpoint(
            #     f"models/{model_name}{fold}_checkpoint", monitor='val_auc', save_best_only=True, mode='max'),
            # EarlyStopping(monitor='val_auc', patience=3, verbose=1,
            #               restore_best_weights=True, mode='max'),
        ]
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # weight, initial_bias = class_weight(train_dataset)
        # print(weight)

        model = densenet_model(learning_rate=learning_rate)
        history = model.fit(X_train, y_train, batch_size=batch_size,
                            epochs=epochs, verbose=1, callbacks=callbacks)

        # plt = plot_metrics(history)
        # experiment.log_figure(
        #     figure=plt, figure_name=f"Metrics History, Fold {fold}")

    model.save(f"models/{model_name}_fold{fold}")

    with experiment.test():
        print("EVALUATE")
        scores = model.evaluate(X_test, y_test, batch_size=batch_size)
        metrics = dict(zip(model.metrics_names, scores))
        print(metrics)
        experiment.log_metrics(metrics, prefix=f"fold{fold}")

        print("MAKE CONFUSION MATRIX")
        y_pred = model.predict(X_test, batch_size=batch_size)
        y_pred = parse_predict(y_pred)

        conf_matrix = confusion_matrix(y_test, y_pred)
        print(conf_matrix)

        plt = plot_confusion_matrix(conf_matrix, ['0', '1'])
        experiment.log_figure(
            figure=plt, figure_name=f"Confusion Matrix, Fold {fold}")

    print("SAVE AND EVALUATE")
    # evaluate_model(model, f"model_predictions/{model_name}_fold{fold}")

    fold = fold + 1

experiment.end()
