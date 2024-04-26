# meMIA Demo Code

This is the demo code for meMIA attack method.

## Building Datasets
We prefer the users could provide the dataloader by themselves. But we show the demo dataloader in the code. Due to the size of the dataset, we won't upload it to github.

For UTKFace, we have two folders downloaded from [official website](https://susanqq.github.io/UTKFace/) in the UTKFace folder. The first is the "processed" folder which contains three landmark_list files(also can be downloaded from the official website). It is used to get the image name in a fast way because all the features of the images can be achieved from the file names. The second folder is the "raw" folder, which contains all the aligned and cropped images. 

For FMNIST and STL10, PyTorch has offered the datasets and they can be easily employed.

## Preparing
Users should install Python3 and PyTorch at first.

## Testing
```python demo.py --attack_type X --dataset_name Y```

<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Attack Type</td>
<td align="center">0</td>
</tr>
<tr>
<td align="center">Name</td>
<td align="center">MemInf</td>
</tr>
</tbody></table>

For dataset name, there are 4 datasets in the code, namely, FMNIST (Fashion-MNIST), STL10, and UTKFace.

For AttrInf, users should provide two attributes in the command line with the format "X_Y" and only CelebA and UTKface contain 2 attributes, e.g. 
```python meMIA_main.py --attack_type 0 --dataset_name purchase --mode -1 --train_shadow --train_model

### For MemInf
We have one mode in this function
<table><tbody>
<!-- TABLE BODY -->
<tr>
<td align="center">Mode</td>
<td align="center">0</td>
</tr>
<tr>
<td align="center">Name</td>
<td align="center">BlackBox Shadow</td>
</tr>
</tbody></table>

