from tensorflow.keras.models import load_model
import tensorflow as tf
from utils import evaluate_model
from models import f1
from tensorflow.keras.optimizers import Adam
dependencies = {
    'f1': f1
}
model_name = "densenet121_v2_fold0"
model = load_model(f"models/{model_name}", custom_objects=dependencies, compile=False)
model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy', f1])
evaluate_model(model, f"model_predictions/{model_name}")
