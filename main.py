from sklearn.model_selection import StratifiedKFold

from models import *
from utils import load_dataset, output_bias
from trainer import evaluate_fold
NFOLDS = 3
INPUT_PATH = 'unibuc-2019-m2-cv'
OUTPUT_PATH = ''

batch_size = 32
epochs = 5
learning_rate = 0.0005
model_name = "densenet121_v4"
model_klass = densenet_model


X, y = load_dataset(train_labels_path=f'{INPUT_PATH}/train_labels.txt',
                    img_path=f'{INPUT_PATH}/data/data')
initial_bias = output_bias(y)

skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True)
split_gen = skf.split(X, y)

fold = 0
for train_index, test_index in split_gen:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    print(f"EVALUATE FOLD {fold}")
    evaluate_fold(model_klass=model_klass,
                  training_data=(X_train, y_train),
                  test_data=(X_test, y_test),
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  epochs=epochs,
                  fold=fold,
                  output_bias=initial_bias,
                  output_path=OUTPUT_PATH,
                  input_path=INPUT_PATH)
    fold += 1
