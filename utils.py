import pandas as pd
import tensorflow as tf

from tf_dataset import AUTOTUNE, process_test_path


def evaluate_model(model, model_name):
    list_ds = tf.data.Dataset.list_files("data/test/*")
    test_ds = list_ds.map(process_test_path, num_parallel_calls=AUTOTUNE)
    y = model.predict(test_ds.batch(32))
    ans_df = pd.DataFrame({'id': [f"{'0'* (6-len(str(x)))}{x}" for x in range(17001, 22150)],
                           'class': parse_predict(y)})
    ans_df.to_csv(f'{model_name}.csv', index=False)


def parse_predict(y_pred):
    return (y_pred.flatten() > 0.5).astype(int)
