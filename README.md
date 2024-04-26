# meMIA Demo Code

This repository contains the demo code for the meMIA (Membership Inference Attack) method.

## Building Datasets
Users are encouraged to provide their own data loaders. However, a demo data loader is included in the code. The datasets themselves are not uploaded to GitHub due to their size.

For the UTKFace dataset, two folders are downloaded from the [official website](https://susanqq.github.io/UTKFace/) into the `UTKFace` directory:

- The `processed` folder contains three `landmark_list` files (also available on the official website). This folder aids in quickly retrieving image names since all image features can be deduced from the filenames.
- The `raw` folder includes all aligned and cropped images.

For the FMNIST and STL10 datasets, PyTorch provides built-in support, making these datasets easy to use.

## Preparing
Before running the demo, users should have Python 3 and PyTorch installed on their system.

## Testing
To test the attack method, use the following command:

```bash
python demo.py --attack_type X --dataset_name Y
