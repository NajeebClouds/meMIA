# meMIA

This repository contains the demo code for the meMIA (Membership Inference Attack) method.

## Building Datasets

### UTKFace Dataset

For the UTKFace dataset, we utilize two folders that can be downloaded from the [official website](https://susanqq.github.io/UTKFace/):

- **processed**: Contains three `landmark_list` files (also downloadable from the official website). These files facilitate quick access to image names, as all image features can be inferred from the file names.
  
- **raw**: Contains all aligned and cropped images from the dataset.

### Other Datasets

- **FMNIST, STL10, CIFART-10, CIFAR-100**: PyTorch provides these datasets, which can be easily used within the framework.

- **Location and Purchase datasets**: These must be downloaded from the following links, and their usage is discussed in the paper:
  - [Foursquare Dataset](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)
  - [Kaggle Acquire Valued Shoppers Challenge Data](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data)

## Setup

Before executing the demo, ensure that Python 3 and PyTorch are installed on your system.

## Testing

To execute the demo code, use the following command:

```bash
# If you have trained shadow and target models, run the following
python meMIA_main.py --attack_type X --dataset_name Y

# If you have not trained shadow and target models, run the following
python meMIA_main.py --attack_type X --dataset_name Y --mode -1 --train_shadow --train_model

# Example
python meMIA_main.py --attack_type 0 --dataset_name cifar10 --mode -1 --train_shadow --train_model
