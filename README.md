# Normalization Flow Model for Anomaly Detetction

# Create a Docker Environment  

```bash 
docker build -t normalization_flow .
```

```bash
docker run --name normalization_flow_run --gpus all -it --rm -v $(pwd):/Normalization_Flow_Test normalization_flow**
```
# Installation Requirements

Before you start, we need to install some requirements folder by running the following:

```bash
pip install -r requirments.txt
```

## Datasets

There are 2 types of dataset which has the pre-built datasetloader which is

1. **MVTec Dataset**
2. **Mold Injection Machine**

The structure of the dataset for the **Mold dataset** folder should be:

```bash
mold_datasets
├── ground_truth
│   ├──anomaly
│   │   ├──001.png
│   │   │  ...
│   │   └──015.png
├── test
│   ├──anomaly
│   └──good
└── train
    └──good
```

Since this is the **unsupervised learning** the labels are not needed. In training the neccessary folder is `mold_datasets/train/good` and the rest are for test purposes.

The **MVTec** dataset folder strcuture are:

```bash
mvtec_datasets
├── pill
├── bottle
├── ...
└── zipper
```

## Training

To start the training, run the following command:

```bash
python main.py --data path/to/dataset/ --category bottle
```

## Evaluation

For do the evaluation, run the following command:

```bash
python main.py --data path/to/dataset/ --category bottle --eval --checkpoint path/to/weight
```
