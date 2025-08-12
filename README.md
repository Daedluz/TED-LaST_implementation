# TED-LaST Implementation

This is an implementation following the paper [TED-LaST](https://arxiv.org/abs/2506.10722), a robust way for detecting poisoned samples to defend against backdoor attacks.

## About This Repository

This is an experimental research implementation.  
The code is functional but may not be fully optimized or extensively documented.

## Project Structure

```
File
├── cache
│   ├── activations.pt
│   │   ├── activations for each layer
│   ├── labels.pt
│   │   ├── labels for the training data
│   ├── knn_{layer_name}.pt
│   │   ├── knn for each layer
├── data
│   ├── cifar10 (or any other dataset you want to use)
│   │   ├── poisoned
│   │   ├── clean
│   │   ├── test
│   │   └── test_poisoned
├── models
│   └── {poisoned_model}.pt

Code
├── resnet.py
│   ├── Definition of poisoned model structure
├── dataset.py
│   ├── Definition of custom dataset object
├── calculate_ctd.py
│   ├── Get activations and calculate ctd function
├── pca.py
│   ├── Train a pca-based outlier detector for each class
├── main.py
│   └── Load datasets, preprocess, and calculate cumulative topological distance
└── utils.py
```

## Getting started

### Dataset Acquisition

This implementation uses datasets and models produced from [Circumventing-Backdoor-Defenses](https://github.com/Unispac/Circumventing-Backdoor-Defenses#), specifically there implementation of adaptive attacks.
Please follow the instructions on the repo to acquire the poisoned dataset and train a target model on that dataset. To acquire poisoned testing samples, I wrote a `make_poison_sample.py` script that you can copy into `Circumventing-Backdoor-Defenses` and execute.
After that, please structure your project as shown above, place the trained model in the `models` directory, and place the poisoned dataset in the `data` directory as specified in the project structure.

### Code Execution

To execute the code, you can run the `main.py` script, which will load the datasets, preprocess the data, and calculate the cumulative topological distance.
Then you can use `pca.py` to train a PCA-based outlier detector for each class and produce anomaly scores for inference samples.
Select the appropriate threshold for the anomaly scores to identify potential backdoor samples.
