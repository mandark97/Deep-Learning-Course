from comet_ml import Experiment
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
from utils import evaluate_model, load_dataset, crossval_ds, get_labels, class_weight, plot_confusion_matrix
from models import *


experiment = Experiment(api_key="i9Sew6Jy0Z36IZaUfJuR0cxhT",
                        project_name="general", workspace="mandark")
NFOLDS = 3
batch_size = 32
epochs = 10
learning_rate = 0.001
model_name = "densenet121_v2"
params = {
    "batch_size": batch_size,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "model_name": model_name,
}
experiment.log_parameters(params)

metrics_arr = []
dataset = load_dataset()
for fold in range(3):
    model = densenet_model(learning_rate=learning_rate)
    print(f"START FOR FOLD {fold}")
    with experiment.train():
        callbacks = [
            ModelCheckpoint(
                f"models/{model_name}{fold}_checkpoint", monitor='f1', save_best_only=True),
        ]
        train_dataset = crossval_ds(
            dataset, n_folds=NFOLDS, val_fold_idx=fold, training=True)
        weight = class_weight(train_dataset)
        model.fit(train_dataset.batch(batch_size),
                  epochs=epochs, verbose=1, callbacks=callbacks,  class_weight=weight)

    with experiment.test():
        print("EVALUATE")
        test_dataset = crossval_ds(
            dataset, n_folds=NFOLDS, val_fold_idx=fold, training=False)
        loss, accuracy, f1_score = model.evaluate(
            test_dataset.batch(batch_size))
        print(loss, accuracy, f1_score)
        metrics = {
            f'loss_fold{fold}': loss,
            f'accuracy_fold{fold}': accuracy,
            f'f1_score_fold{fold}': f1_score
        }
        metrics_arr.append(metrics)
        experiment.log_metrics(metrics)

        print("MAKE CONFUSION MATRIX")
        y_true = get_labels(test_dataset)
        y_pred = np.argmax(
            model.predict(test_dataset.batch(batch_size)),
            axis=1)

        conf_matrix = confusion_matrix(y_true, y_pred)
        print(conf_matrix)
        plt = plot_confusion_matrix(conf_matrix, ['0', '1'])
        experiment.log_figure(
            figure=plt, figure_name=f"Confusion Matrix, Fold {fold}")

        # experiment.log_confusion_matrix(matrix=conf_matrix,
        #                                 title=f"Confusion Matrix, Fold {fold}",
        #                                 file_name=f"conf_matrix{fold}.json")
    print("SAVE AND EVALUATE")
    model.save(f"models/{model_name}_fold{fold}")
    evaluate_model(model, f"model_predictions/{model_name}_fold{fold}")

experiment.end()
