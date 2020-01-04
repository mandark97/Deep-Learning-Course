import json

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.metrics import (AUC, BinaryAccuracy, FalseNegatives, FalsePositives,
                           Precision, Recall, TrueNegatives, TruePositives)
from sklearn.metrics import confusion_matrix

from utils import (parse_predict, plot_confusion_matrix, plot_metrics,
                   run_test_data)


def f1(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    return 2 * (K.sum(y_true * y_pred) + K.epsilon()) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


def build_model(model_klass, learning_rate, output_bias=None):
    K.clear_session()
    metrics = [
        TruePositives(name='tp'),
        FalsePositives(name='fp'),
        TrueNegatives(name='tn'),
        FalseNegatives(name='fn'),
        BinaryAccuracy(name='accuracy'),
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc'),
        f1
    ]

    model = model_klass(learning_rate=learning_rate,
                        metrics=metrics, output_bias=output_bias)

    return model


def callbacks(checkpoint_path):
    return [
        ModelCheckpoint(checkpoint_path, monitor='val_auc',
                        save_best_only=True, mode='max'),
        EarlyStopping(monitor='val_auc', patience=3, verbose=1,
                      restore_best_weights=True, mode='max'),
    ]


def save_metrics(scores, metrics_names, output_path):
    metrics = dict(zip(metrics_names, scores))
    print(metrics)
    with open(output_path, 'w') as json_file:
        json.dump(metrics, json_file)


def evaluate_fold(model_klass, training_data, test_data, batch_size, learning_rate, epochs, fold, input_path, output_path, output_bias=None):
    model = build_model(model_klass, learning_rate, output_bias)

    X_train, y_train = training_data
    history = model.fit(X_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.15,
                        verbose=1,
                        callbacks=callbacks(checkpoint_path=f'{output_path}/{fold}_checkpoint.h5'))
    model.save(f'{output_path}/fold{fold}.h5')
    plot_metrics(history, savefig=f'{output_path}/metrics{fold}.png')

    X_test, y_test = test_data
    scores = model.evaluate(X_test, y_test, batch_size=batch_size)
    save_metrics(scores, model.metrics_names,
                 output_path=f'{output_path}/metrics{fold}.json')

    y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = parse_predict(y_pred)

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    plot_confusion_matrix(conf_matrix, ['0', '1'],
                          title=f'Confusion Matrix fold {fold}',
                          savefig=f'{output_path}/conf_matrix_fold{fold}.png')

    run_test_data(model, f'{output_path}/{fold}.csv',
                  img_path=f'{input_path}/data/data')
