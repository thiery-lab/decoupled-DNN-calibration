import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision
import pickle
import numpy as np
from PIL import Image

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
            self.img_dir = root + 'train_images/'
            label_csv = root + 'trainLabels.csv'
        else:
            self.img_dir = root + 'test_images/'
            label_csv = root + 'testLabels.csv'
            
        self.transform = transform
        self.binary = binary
        with open(label_csv, 'r') as label_file:
            label_tuple = [line.strip().split(',')[:2] for line in label_file.readlines()[1:]]
            self.imgs = [item[0] for item in label_tuple]
            self.labels = [int(item[1]) for item in label_tuple]
            bad_images = ['10_left']
            for img in bad_images:
                if img in self.imgs:
                    index = self.imgs.index(img)
                    self.imgs = self.imgs[:index] + self.imgs[index+1:]
                    self.labels = self.labels[:index] + self.labels[index+1:]


    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image = Image.open(self.img_dir + img_name + '.jpeg')
        image = image.resize((256, 256), resample=Image.BILINEAR)
        image = self.transform(image)
        label = self.labels[idx]
        if self.binary:
            label = min(label, 1)
        return image, label


class encoded_cifar10(data.Dataset):
    'Characterizes a dataset for PyTorch'


    def __init__(self, root, train, depth):
        'Initialization'
        
        self.feature = np.array([])
        self.target = np.array([])
        
        if depth==110:
            file_start = 'encoded110ResNet_CIFAR10_256'
        else:
            file_start = 'encoded164ResNet_CIFAR10_256'
        
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


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.target)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Return feature, label at index
        X = torch.tensor(self.feature[index], dtype=torch.float)
        y = torch.tensor(self.target[index])
        
        return X, y
                
                
class wideresnet_encoded_cifar10(data.Dataset):
    'Characterizes a dataset for PyTorch'


    def __init__(self, root, train, depth):
        'Initialization'
        
        self.feature = np.array([])
        self.target = np.array([])
        
        if depth==28:
            file_start = 'encoded28x10WideResNet_CIFAR10_640'
        else:
            file_start = ''
        
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

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.target)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Return feature, label at index
        X = torch.tensor(self.feature[index], dtype=torch.float)
        y = torch.tensor(self.target[index])
        
        return X, y
                
                
class encoded_cifar100(data.Dataset):
    'Characterizes a dataset for PyTorch'


    def __init__(self, root, train, depth):
        'Initialization'
        
        self.feature = np.array([])
        self.target = np.array([])
        
        if depth==110:
            file_start = 'encoded110ResNet_CIFAR100_256'
        else:
            file_start = 'encoded164ResNet_CIFAR100_256'
        
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

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.target)


    def __getitem__(self, index):
        'Generates one sample of data'
        # Return feature, label at index
        X = torch.tensor(self.feature[index], dtype=torch.float)
        y = torch.tensor(self.target[index])
        
        return X, y


DATASET_PATH = {'CIFAR10' : '/home/rahul/lab_work/data/',
                'CIFAR100' : '/home/rahul/lab_work/data/',
                'ENCODED256_CIFAR10' : '/home/rahul/lab_work/data/encoded_cifar/',
                'ENCODED256_CIFAR100' : '/home/rahul/lab_work/data/encoded_cifar/',
                'DIABETIC_RETINOPATHY' : '/data02/DIABETIC_RETINOPATHY/'
               }


DATASET_DICTIONARY = {'CIFAR10_TRAIN' : {'class_function' : torchvision.datasets.CIFAR10,
                                        'init_params' : {'root' : DATASET_PATH['CIFAR10'],
                                                        'train' : True,
                                                        'download' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))])
                                                        }
                                        },
                      'CIFAR10_TEST' : {'class_function' : torchvision.datasets.CIFAR10,
                                        'init_params' : {'root' : DATASET_PATH['CIFAR10'],
                                                        'train' : False,
                                                        'download' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))])
                                                        }
                                        },
                     'CIFAR100_TRAIN' : {'class_function' : torchvision.datasets.CIFAR100,
                                        'init_params' : {'root' : DATASET_PATH['CIFAR100'],
                                                        'train' : True,
                                                        'download' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))])
                                                        }
                                        },
                      'CIFAR100_TEST' : {'class_function' : torchvision.datasets.CIFAR100,
                                        'init_params' : {'root' : DATASET_PATH['CIFAR100'],
                                                        'train' : False,
                                                        'download' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))])
                                                        }
                                        },
                     'ENCODED256_D110_CIFAR10_TRAIN' : {'class_function' : encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 0,
                                                                   'depth' : 110
                                                                  }
                                                  },
                     'ENCODED256_D110_CIFAR10_VALID' : {'class_function' : encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 2,
                                                                   'depth' : 110
                                                                  }
                                                  },
                     'ENCODED256_D110_CIFAR10_TEST' : {'class_function' : encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 1,
                                                                   'depth' : 110
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR10_TRAIN' : {'class_function' : encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 0,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR10_VALID' : {'class_function' : encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 2,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR10_TEST' : {'class_function' : encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR100_TRAIN' : {'class_function' : encoded_cifar100,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 0,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR100_VALID' : {'class_function' : encoded_cifar100,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR100_TEST' : {'class_function' : encoded_cifar100,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED640_D28_CIFAR10_TRAIN' : {'class_function' : wideresnet_encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 0,
                                                                   'depth' : 28
                                                                  }
                                                  },
                     'ENCODED640_D28_CIFAR10_VALID' : {'class_function' : wideresnet_encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 2,
                                                                   'depth' : 28
                                                                  }
                                                  },
                     'ENCODED640_D28_CIFAR10_TEST' : {'class_function' : wideresnet_encoded_cifar10,
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 1,
                                                                   'depth' : 28
                                                                  }
                                                  },
                     'DIAB_RETIN_TRAIN' : {'class_function' : retinopathy_dataset,
                                        'init_params' : {'root' : DATASET_PATH['DIABETIC_RETINOPATHY'],
                                                        'train' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))]),
                                                        'binary' : True
                                                        }
                                        },
                      'DIAB_RETIN_TEST' : {'class_function' : retinopathy_dataset,
                                        'init_params' : {'root' : DATASET_PATH['DIABETIC_RETINOPATHY'],
                                                        'train' : False,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))]),
                                                        'binary' : True
                                                        }
                                        },
                     }


def generate_dataloaders(data_name, batch_size, shuffle, num_workers, root=None):
    '''
        generates dataloaders and returns it. Data name is one of the keys in
        DATASET_DICTIONARY which is stored in this module 'data_utils' e.g: 'CIFAR10_TRAIN'
        batch_size, shuffle, num_workers are pytorch dataloader parameters
    '''
    
    dataset_attributes = DATASET_DICTIONARY[data_name]
    if root is not None:
        dataset_attributes['init_params']['root'] = root
    dataset = dataset_attributes['class_function'](**dataset_attributes['init_params'])
    
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
