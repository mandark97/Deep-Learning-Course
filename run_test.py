from tensorflow.keras.models import load_model
import tensorflow as tf

model = load_model("ceva")
list_ds = tf.data.Dataset.list_files("data/test/*/*")
y = model.predict(list_ds)
print(y)
