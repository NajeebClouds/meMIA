# meMIA

This repository contains the demo code for the meMIA (Membership Inference Attack) method.

## Building Datasets


### UTKFace Dataset

For the UTKFace dataset, we utilize two folders that can be downloaded from the [official website](https://susanqq.github.io/UTKFace/):

- `processed`: This folder contains three `landmark_list` files (also downloadable from the official website). These files are used for quickly accessing image names, as all image features can be inferred from the file names.
- `raw`: Contains all aligned and cropped images from the dataset.

### Other Datasets

For FMNIST and STL10, CIFART-10, CIFAR-100 PyTorch provides these datasets, which can be easily used within the framework.

## Setup

Before executing the demo, make sure Python 3 and PyTorch are installed on your system.

## Testing

Execute the demo code using the following command:
| Attack Type | Name   |
|-------------|--------|
| 0           | MemInf |
```bash
#If you have trained shadow and target models, run the following
python meMIA_main.py --attack_type X --dataset_name Y


#If you have note trained shadow and target models, run the following
python meMIA_main.py --attack_type X --dataset_name Y --mode -1 --train_shadow --train_model

# Example
python meMIA_main.py --attack_type 0 --dataset_name cifar10 --mode -1 --train_shadow --train_model



