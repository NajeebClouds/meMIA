import os
import torch
import pandas
import torchvision
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.autograd import Variable
from torch.utils.data import random_split, ConcatDataset
from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple




class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
class simpleNN(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(128, num_classes),
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x
    
class simpleNN_Target_purchase(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN_Target_purchase, self).__init__()
        
        self.classifier = nn.Sequential(
            
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),      
            nn.Linear(100, num_classes),
            
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x

class simpleNN_Shaddow_purchase(nn.Module):
    def __init__(self, input_size, num_classes=30):
        super(simpleNN_Shaddow_purchase, self).__init__()
        
        self.classifier = nn.Sequential(
            
            nn.Linear(input_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),      
            nn.Linear(100, num_classes),
       
            
        )


    def forward(self, X_Batch):
        x = self.classifier(X_Batch)
        return x



class UTKFaceDataset(torch.utils.data.Dataset):
    
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        self.processed_path = os.path.join(self.root, 'UTKFace/processed/')
        
        
        self.files = os.listdir(self.processed_path)
        print("self.root: ", self.root)    
        print("in the UTKFace dataset class constructor", self.processed_path)
        print("self files: ", self.files)
        # exit()
        
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            txt_file_path = os.path.join(self.processed_path, txt_file)
            with open(txt_file_path, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4  or '' in attrs:
                        continue
                    self.lines.append(image_name+'jpg')


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        attrs = self.lines[index].split('_')

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])
        # print("in the __getitem__ method")
        
        image_path = os.path.join(self.root, 'UTKFace/raw/', self.lines[index]+'.chip.jpg').rstrip()
        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target



def Location(num_classes):
    
    dataset = []
    locations = []
    labels = []
    output_coll =  torch.empty((0, num_classes))
    
    file_path = "data/location/bangkok"

    if os.path.exists(file_path):
        print(f"yes")
    else:
        print(f"No.")
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Split the line by commas
            if len(parts) >= 2:
                # Convert the first part to an integer (label)
                cleaned_string = parts[0].replace('"', '')
                label = int(cleaned_string)
                location = [int(token) for token in parts[1:]]  # Convert the rest to integers (locations)
                # print(len(location))
                labels.append(label)
                locations.append(location)
            # break

    # Convert lists to NumPy arrays
    Y = np.array(labels)
    Y = Y.reshape(-1, 1)
    Y = Y - 1
    # print(labels_array.shape)
    X = torch.tensor(locations, dtype=torch.float)
   
    
    for i in range(X.size()[0]):
        dataset.append((X[i], Y[i].item()))
    
    
    return dataset


def Purchase(num_classes):
    
    
    dataset = []
    purchase_feats = []
    purchase_labels = []
    output_coll =  torch.empty((0, num_classes))
    

    file_path = "data/purchase/purchase"
    if os.path.exists(file_path):
        print(f"yes")
    else:
        print(f"No.")
    
    # exit()
    
    # Read the file line by line
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')  # Split the line by commas
            if len(parts) >= 2:
               
                purchase_label = int(parts[0])
                # print(f"purchase_label: {type(purchase_label)}")
                
                purchase_feat = [int(token) for token in parts[1:]]  # Convert the rest to integers (locations)
                # print(len(purchase_feat))
                purchase_labels.append(purchase_label)
                purchase_feats.append(purchase_feat)
                
            # break

    # Convert lists to NumPy arrays
    Y = np.array(purchase_labels)
    Y = Y.reshape(-1, 1)
    Y = Y-1
    
    X = torch.tensor(purchase_feats, dtype=torch.float)
   
    for i in range(X.size()[0]):
        dataset.append((X[i], Y[i].item()))
    
    
    print(f"dataSet size: {Y[:19]}")
    print(f"type {type(dataset)}")
    print(f"size of dataset: {len(dataset)}")
    
    # exit()

    return dataset


def prepare_dataset(dataset, attr, root, device):
    
    num_classes, dataset, target_model, shadow_model = get_model_dataset(dataset, device, attr=attr, root=root)
    length = len(dataset)
    
    traning_size = 2500    
    split_len = traning_size//2
    
    each_length = length//4
    
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [split_len, split_len, split_len, split_len, len(dataset)-(split_len*4)])
    
    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model



def get_model_dataset(dataset_name, device, attr, root):
    print(f"device in get_model_dataset: {device}")
    # exit()
    if dataset_name.lower() == "utkface":
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        print("attributes: ", attr)
        
        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)
        input_channel = 3
        
    elif dataset_name.lower() == "celeba":
        if isinstance(attr, list):
            for a in attr:
                if a != "attr":
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))

                num_classes = [8, 4]
                # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                attr_list = [[18, 21, 31], [20, 39]]
        else:
            if attr == "attr":
                num_classes = 8
                attr_list = [[18, 21, 31]]
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CelebA(root=root, attr_list=attr_list, target_type=attr, transform=transform)
        input_channel = 3

    elif dataset_name.lower() == "stl10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.STL10(
                root=root, split='train', transform=transform, download=True)
            
        test_set = torchvision.datasets.STL10(
                root=root, split='test', transform=transform, download=True)
        
        dataset = train_set + test_set
        input_channel = 3
        print(f"size of STL10 dataset: {len(train_set), len(test_set)}")
        
        # exit()
    
    elif dataset_name.lower() == "cifar10":
        
        print(f"CIFAR10")
        
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_set = torchvision.datasets.CIFAR10(
                root=root, train=True, transform=transform, download=True)
        test_set = torchvision.datasets.CIFAR10(
                root=root, train=False, transform=transform, download=True)

        dataset = train_set + test_set
        input_channel = 3
        print(f"size of CIFAR10 dataset: {len(train_set), len(test_set)} and T dataSize: {len(dataset)}")
        img, label = dataset[0]
        # print(dataset[1])
        print(f"type of cifar dataset: {type(train_set)}")
        # exit()
    
    elif dataset_name.lower() == "cifar100":
        
        print(f"CIFAR10")
        
        num_classes = 100
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_set = torchvision.datasets.CIFAR100(
                root=root, train=True, transform=transform, download=True)
        test_set = torchvision.datasets.CIFAR100(
                root=root, train=False, transform=transform, download=True)

        dataset = train_set + test_set
        input_channel = 3
        print(f"size of CIFAR100 dataset: {len(train_set), len(test_set), len(dataset)}")
        # exit()
    
    elif dataset_name.lower() == "location":
        
        print(f"Location dataset")
        
        num_classes = 30
        dataset = Location(num_classes)
        first_sample, _ = dataset[0]
        input_size = len(first_sample)
        print(f"size of location dataset: {len(first_sample)}")

    elif dataset_name.lower() == "purchase":
        num_classes = 100
        print(f"purchase dataset")
        dataset = Purchase(num_classes)
        first_sample, _ = dataset[0]
        input_size = len(first_sample)
        print(f"input size of purchase dataset: {input_size}")
        # exit()
    
    elif dataset_name.lower() == "fmnist":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = torchvision.datasets.FashionMNIST(
                root=root, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.FashionMNIST(
                root=root, train=False, download=True, transform=transform)

        dataset = train_set + test_set
        input_channel = 1
        print(f"size of FMNIST dataset: {len(train_set), len(test_set)}")
        # exit()
        

    if isinstance(num_classes, int):
        # print(f"isininstances if, device: {device}")
        
        if dataset_name.lower() == "location":
            target_model = simpleNN(input_size = input_size, num_classes=num_classes)
            shadow_model = simpleNN(input_size = input_size, num_classes=num_classes)
        elif dataset_name.lower() == "purchase":
            target_model = simpleNN_Target_purchase(input_size = input_size, num_classes=num_classes)
            shadow_model = simpleNN_Shaddow_purchase(input_size = input_size, num_classes=num_classes)
        else:
            target_model = CNN(input_channel=input_channel, num_classes=num_classes)
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes)
      
        
    else:
        if dataset_name.lower() == "location":
            target_model = simpleNN(input_size = input_size, num_classes=num_classes[0])
            shadow_model = simpleNN(input_size = input_size, num_classes=num_classes[0])
        elif dataset_name.lower() == "purchase":
            target_model = simpleNN_Target_purchase(input_size = input_size, num_classes=num_classes[0])
            shadow_model = simpleNN_Shaddow_purchase(input_size = input_size, num_classes=num_classes[0])
        else:
            target_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes[0])  
       
    return num_classes, dataset, target_model, shadow_model
