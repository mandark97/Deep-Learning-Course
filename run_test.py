from keras.models import load_model
from keras.optimizers import Adam

from utils import run_test_data


def run_from_save(model_path, img_path, output_path):
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    run_test_data(model, output_path,
                  img_path=img_path)


# run_from_save("resnet101_model_v2_fold2")
