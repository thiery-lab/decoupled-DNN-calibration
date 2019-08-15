import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import pickle
import numpy as np
from PIL import Image
from .data_config import *
import sys

class retinopathy_dataset(data.Dataset):
    '''DIABETIC RETINOPATHY dataset downloaded from
    kaggle link : https://www.kaggle.com/c/diabetic-retinopathy-detection/data
    root : location of data files
    train : whether training dataset (True) or test (False)
    transform : torch image transformations
    binary : whether healthy '0' vs damaged '1,2,3,4' binary detection (True) 
             or multiclass (False)
    '''


    def __init__(self, root, train, transform, binary=True, balance=True):
        if train:
            self.img_dir = root + 'train/'
            label_csv = root + 'trainLabels.csv'
        else:
            self.img_dir = root + 'test/'
            label_csv = root + 'testLabels.csv'
            
        self.transform = transform
        self.binary = binary
        with open(label_csv, 'r') as label_file:
            label_tuple = [line.strip().split(',')[:2] for line in label_file.readlines()[1:]]
        self.imgs = [item[0] for item in label_tuple]
        self.labels = [int(item[1]) for item in label_tuple]
        if self.binary:
            self.labels = [min(label,1) for label in self.labels]

        bad_images = ['10_left']
        for img in bad_images:
            if img in self.imgs:
                index = self.imgs.index(img)
                self.imgs = self.imgs[:index] + self.imgs[index+1:]
                self.labels = self.labels[:index] + self.labels[index+1:]

        # make all these better
        classes, counts = np.unique(np.array(self.labels), return_counts=True)
        deviation = np.std(counts) / np.mean(counts)
        if deviation > 0.05 and train:
            weights = 1./torch.tensor(counts, dtype=torch.float)
            self.sample_weights = weights[self.labels]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image = Image.open(self.img_dir + img_name + '.jpeg')
        image = image.resize((256, 256), resample=Image.BILINEAR)
        image = self.transform(image)
        label = self.labels[idx]

        return image, label

    
class encoded_dataset(data.Dataset):
    """
    Base class for encoded dataset
    """
    def __init__(self, root, train, depth, withGP=False):
        'Initialization'
        
        self.feature = np.array([])
        self.target = np.array([])
        file_start = self.generate_file_name(depth, withGP)
        
        if train==0:
            for file_count in range(5):
                with open(root + file_start + '.pt' + str(file_count+1), 'rb') as read_file:
                    data = pickle.load(read_file)
                    if len(self.feature) == 0:
                        self.feature = data['feature']
                        self.target = data['label']
                    else:
                        self.feature = np.concatenate((self.feature, data['feature']), axis=0)
                        self.target = np.concatenate((self.target, data['label']), axis=0)
        elif train==1:
            with open(root + file_start + '_test', 'rb') as read_file:
                data = pickle.load(read_file)
                self.feature = data['feature']
                self.target = data['label']
        else:
            with open(root + file_start + '_valid', 'rb') as read_file:
                data = pickle.load(read_file)
                self.feature = data['feature']
                self.target = data['label']

    
    def generate_file_name(self, depth, withGP):
        """
        generate base file name for the encoded features
        """
        raise NotImplementedError
        

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.target)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Return feature, label at index
        X = torch.tensor(self.feature[index], dtype=torch.float)
        y = torch.tensor(self.target[index])
        
        return X, y
    
    
class resnet_encoded_cifar10(encoded_dataset):
    'Characterizes a dataset for PyTorch'

    def generate_file_name(self, depth, withGP):
        file_start = ''
        if withGP:
            file_start += 'GP+'
        file_start += 'encoded' + str(depth) + 'ResNet_CIFAR10_256'
        return file_start


class wideresnet_encoded_cifar10(encoded_dataset):
    'Characterizes a dataset for PyTorch'

    def generate_file_name(self, depth, withGP):
        file_start = ''
        if depth==28:
            depth = '28x10'
        if withGP:
            file_start += 'GP+'
        file_start += 'encoded' + str(depth) + 'WideResNet_CIFAR10_256'
        return file_start
    
                
class resnet_encoded_cifar100(encoded_dataset):
    'Characterizes a dataset for PyTorch'

    def generate_file_name(self, depth, withGP):
        file_start = ''
        if withGP:
            file_start += 'GP+'
        file_start += 'encoded' + str(depth) + 'ResNet_CIFAR100_256'
        return file_start


def generate_dataloaders(data_name, batch_size, shuffle, num_workers, root=None):
    '''
        generates dataloaders and returns it. Data name is one of the keys in
        DATASET_DICTIONARY which is stored in this module 'data_utils' e.g: 'CIFAR10_TRAIN'
        batch_size, shuffle, num_workers are pytorch dataloader parameters
    '''
    
    dataset_attributes = DATASET_DICTIONARY[data_name]
    if root is not None:
        dataset_attributes['init_params']['root'] = root
    
    dataset_class = dataset_attributes['class_function']
    if isinstance(dataset_class, str):
        dataset_class = getattr(sys.modules[__name__], dataset_class)
    dataset = dataset_class(**dataset_attributes['init_params'])

    if hasattr(dataset, 'sample_weights'):
        print("Weighted sampler has been selected to balance classes")
        sampler = data.WeightedRandomSampler(
            weights=dataset.sample_weights,
            num_samples=len(dataset.sample_weights),
            replacement=True)
        return data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=num_workers)
    else:
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
