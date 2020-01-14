import json
import os
from collections import defaultdict
from utils import run_test_data, evaluate_folds


def run_from_save(model_path, img_path, output_path):
    from keras.models import load_model
    from keras.optimizers import Adam
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    run_test_data(model, output_path,
                  img_path=img_path)


def get_results():
    for dir1 in os.listdir("results"):
        for dir2 in os.listdir(f"results/{dir1}"):
            results = defaultdict(list)

            for i in range(3):
                with open(f"results/{dir1}/{dir2}/metrics{i}.json", 'r') as file:
                    metric = json.load(file)
                    for k, v in metric.items():
                        results[k].append(v)

            avg_results = {k: sum(v)/len(v) for k, v in results.items()}

            with open(f"results/{dir1}/{dir2}/results.json", 'w') as out_file:
                json.dump(avg_results, out_file)
