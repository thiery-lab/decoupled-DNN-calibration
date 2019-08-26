import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision


DATASET_PATH = {'CIFAR10' : '/home/rahul/lab_work/data/',
                'CIFAR100' : '/home/rahul/lab_work/data/',
                'ENCODED256_CIFAR10' : '/home/rahul/lab_work/data/',
                'ENCODED256_CIFAR100' : '/home/rahul/lab_work/data/',
                'DIABETIC_RETINOPATHY' : '/data02/'
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
                     'ENCODED256_D110_CIFAR10_TRAIN' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 0,
                                                                  'depth' : 110
                                                                  }
                                                  },
                     'ENCODED256_D110_CIFAR10_VALID' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 2,
                                                                   'depth' : 110
                                                                  }
                                                  },
                     'ENCODED256_D110_CIFAR10_TEST' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 1,
                                                                   'depth' : 110
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR10_TRAIN' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 0,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR10_VALID' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 2,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR10_TEST' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR100_TRAIN' : {'class_function' : 'resnet_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 0,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR100_VALID' : {'class_function' : 'resnet_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED256_D164_CIFAR100_TEST' : {'class_function' : 'resnet_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'ENCODED640_D28_CIFAR10_TRAIN' : {'class_function' : 'wideresnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 0,
                                                                   'depth' : 28
                                                                  }
                                                  },
                     'ENCODED640_D28_CIFAR10_VALID' : {'class_function' : 'wideresnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 2,
                                                                   'depth' : 28
                                                                  }
                                                  },
                     'ENCODED640_D28_CIFAR10_TEST' : {'class_function' : 'wideresnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 1,
                                                                   'depth' : 28
                                                                  }
                                                  },
                     'DIAB_RETIN_TRAIN' : {'class_function' : 'retinopathy_dataset',
                                        'init_params' : {'root' : DATASET_PATH['DIABETIC_RETINOPATHY'],
                                                        'train' : True,
                                                        'transform' : transforms.Compose([transforms.RandomResizedCrop(256),
                                                                                          transforms.RandomHorizontalFlip(),
                                                                                          transforms.ToTensor(),
                                                                                          transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                                         (0.2023, 0.1994, 0.2010))]),
                                                        'binary' : True
                                                        }
                                        },
                      'DIAB_RETIN_TEST' : {'class_function' : 'retinopathy_dataset',
                                        'init_params' : {'root' : DATASET_PATH['DIABETIC_RETINOPATHY'],
                                                        'train' : False,
                                                        'transform' : transforms.Compose([transforms.RandomResizedCrop(256),
                                                                                          transforms.RandomHorizontalFlip(),
                                                                                          transforms.ToTensor(),
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))]),
                                                        'binary' : True
                                                        }
                                        },
                     'GP+ENCODED256_D164_CIFAR10_VALID' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 2,
                                                                   'depth' : 164,
                                                                   'withGP' : True
                                                                  }
                                                      },
                     'GP+ENCODED256_D164_CIFAR10_TEST' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 1,
                                                                   'depth' : 164,
                                                                   'withGP' : True
                                                                  }
                                                      },
                     'GP+ENCODED256_D164_CIFAR10_TRAIN' : {'class_function' : 'resnet_encoded_cifar10',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                  'train' : 0,
                                                                   'depth' : 164,
                                                                   'withGP' : True
                                                                  }
                                                      },
                     'GP+ENCODED256_D164_CIFAR100_VALID' : {'class_function' : 'resnet_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 2,
                                                                   'depth' : 164,
                                                                   'withGP' : True
                                                                  }
                                                      },
                     'GP+ENCODED256_D164_CIFAR100_TEST' : {'class_function' : 'resnet_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                   'depth' : 164,
                                                                   'withGP' : True
                                                                  }
                                                      },
                     'GP+ENCODED256_D164_CIFAR100_TRAIN' : {'class_function' : 'resnet_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 0,
                                                                   'depth' : 164,
                                                                   'withGP' : True
                                                                  }
                                                      }
                     }

