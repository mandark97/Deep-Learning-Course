# Brain Hemorrhage Identification

Identify bleeding in CT scans of the brain

## Requirements

- python3.6
- keras
- tensorflow
- matplotlib
- sklearn
- numpy
- pandas
- opencv


## How to adjust parameters

In `main.py` you can set INPUT_PATH, OUTPUT_PATH, batch_size, epochs, learning_rate, model_name, model_klass, use_output_bias, use_class_weight parameters in order to experiment with different configurations.

`model_klass` parameter represents the architecture of the network. Currently only DenseNet121 and ResNet101 are implemented.
