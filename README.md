![Python](https://img.shields.io/badge/python-3.11.13-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.8.0-orange)
![TorchVision](https://img.shields.io/badge/torchvision-0.23.0-red)
![TorchMetrics](https://img.shields.io/badge/torchmetrics-1.8.2-green)
![CUDA](https://img.shields.io/badge/CUDA-12.8-lightgrey)

# Simple CNN of Cifar10
TinyVGG-inspired CNN architecture for solving Cifar10. 


## Design
Baseline Mode: TinyVGG with data normalisation and early stopping.

Additional experiments with
- Data augmentation
- Batch normalisation
- Dropout

Models validation results are compared, the best model is selected and evaluated on the test set.

- train_model.py
- compare_runs.ipynb
- final_results.ipynb

## Results

## Improvements
Experiment more:
- Batch size
- Hidden units
Implementation ideas:
- Learning rate scheduling
- Continued training from saved checkpoints of the most promising models.
- Other model architectures, e.g., miniResNet


# Running Custom Experiments
## ENV
The env used for developing this project was managed by Conda and is provided in `env.lock.yaml`. You can also clone the repo on a managed platform like Colab or replicate the dependencies manually.

## Workflow
- Define your run/experiment. Use the provided `.yaml` configuration files for reference.
- Run the `train_model.py` script with the appropriate configuration file. e.g.
```bash
python train_model.py --config ./run_config/run_1.yaml
```
- this will save model metrics into the `metrics` directory and save model checkpoints into the `checkpoints` directory.
- you can then compare multiple runs with `compare_runs.ipynb`, which will read the metrics from the `metrics` directory and plot the results.
- choose the best model according to the validation score and evaluate it on the test set with `final_results.ipynb`.