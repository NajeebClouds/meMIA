# meMIA Demo Code

This repository contains the demo code for the meMIA (Membership Inference Attack) method.

## Building Datasets

Users are recommended to provide their own data loaders. However, we include a demo data loader in the code. Due to the large size of the datasets, we do not upload them to GitHub.

### UTKFace Dataset

For the UTKFace dataset, we utilize two folders that can be downloaded from the [official website](https://susanqq.github.io/UTKFace/):

- `processed`: This folder contains three `landmark_list` files (also downloadable from the official website). These files are used for quickly accessing image names, as all image features can be inferred from the file names.
- `raw`: Contains all aligned and cropped images from the dataset.

### FMNIST and STL10 Datasets

For FMNIST and STL10, PyTorch provides these datasets, which can be easily used within the framework.

## Setup

Before executing the demo, make sure Python 3 and PyTorch are installed on your system.

## Testing

Execute the demo code using the following command:
| Attack Type | Name   |
|-------------|--------|
| 0           | MemInf |
```bash
python meMIA_main.py --attack_type X --dataset_name Y


## Exaple for tesing purchase
```bash
python meMIA_main.py --attack_type X --dataset_name Y


