import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision


DATASET_PATH = {'CIFAR10' : '/home/rahul/lab_work/data/',
                'CIFAR100' : '/home/rahul/lab_work/data/',
                'ENCODED256_CIFAR10' : '/home/rahul/lab_work/data/encoded_cifar/',
                'ENCODED256_CIFAR100' : '/home/rahul/lab_work/data/encoded_cifar/',
                'DIABETIC_RETINOPATHY' : '/data02/',
                'ENCODED_DR' : '/home/rahul/lab_work/data/encoded_DR/'
               }


DATASET_DICTIONARY = {'CIFAR10_TRAIN' : {'class_function' : 'reformed_CIFAR10',
                                        'init_params' : {'root' : DATASET_PATH['CIFAR10'],
                                                        'train' : True,
                                                        'download' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))])
                                                        }
                                        },
                      'CIFAR10_TEST' : {'class_function' : 'reformed_CIFAR10',
                                        'init_params' : {'root' : DATASET_PATH['CIFAR10'],
                                                        'train' : False,
                                                        'download' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))])
                                                        }
                                        },
                     'CIFAR100_TRAIN' : {'class_function' : 'reformed_CIFAR100',
                                        'init_params' : {'root' : DATASET_PATH['CIFAR100'],
                                                        'train' : True,
                                                        'download' : True,
                                                        'transform' : transforms.Compose([transforms.ToTensor(), 
                                                                                    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                                                                                         (0.2023, 0.1994, 0.2010))])
                                                        }
                                        },
                      'CIFAR100_TEST' : {'class_function' : 'reformed_CIFAR100',
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
#                                            'collate_fn' : 'diab_retin_collate_fn',
                                        'init_params' : {'root' : DATASET_PATH['DIABETIC_RETINOPATHY'],
                                                        'train' : True,
                                                        'transform' : transforms.Compose([transforms.Resize((512, 512)),  # RandomResizedCrop(512),
                                                                                          #transforms.RandomAffine((0,360)),
                                                                                          #transforms.ColorJitter(brightness=(0.7, 1.3),
                                                                                          #                       contrast=(0.7, 1.3)),
                                                                                          #transforms.RandomHorizontalFlip(),
                                                                                          transforms.ToTensor(),
<<<<<<< HEAD
#                                                                                           transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                                                                                                (0.2023, 0.1994, 0.2010))
=======
                                                                                          # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                          #                      (0.2023, 0.1994, 0.2010))
>>>>>>> 068105f422642bb0e2637a060e741652a7b19c25
                                                                                          ]),
                                                        'binary' : True
                                                        }
                                        },
                      'DIAB_RETIN_TEST' : {'class_function' : 'retinopathy_dataset',
                                           'collate_fn' : 'diab_retin_collate_fn',
                                        'init_params' : {'root' : DATASET_PATH['DIABETIC_RETINOPATHY'],
                                                        'train' : False,
                                                        'transform' : transforms.Compose([transforms.Resize((512, 512)),
                                                                                          transforms.ToTensor(),
                                                                                          # transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                          #                (0.2023, 0.1994, 0.2010))
                                                                                          ]),
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
                                                      },
                     'CIFAR10ENCODED256_D164_CIFAR100_TRAIN' : {'class_function' : 'resnet10_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 0,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'CIFAR10ENCODED256_D164_CIFAR100_VALID' : {'class_function' : 'resnet10_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'CIFAR10ENCODED256_D164_CIFAR100_TEST' : {'class_function' : 'resnet10_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'CIFAR10GPENCODED256_D164_CIFAR100_TRAIN' : {'class_function' : 'resnet10_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 0,
                                                                  'depth' : 164,
                                                                  'withGP' : True
                                                                  }
                                                  },
                     'CIFAR10GPENCODED256_D164_CIFAR100_VALID' : {'class_function' : 'resnet10_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                  'depth' : 164,
                                                                  'withGP' : True
                                                                  }
                                                  },
                     'CIFAR10GPENCODED256_D164_CIFAR100_TEST' : {'class_function' : 'resnet10_encoded_cifar100',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR100'],
                                                                  'train' : 1,
                                                                  'depth' : 164,
                                                                  'withGP' : True
                                                                  }
                                                  },
                     '48000RESNET164_CIFAR10_TRAIN' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                   'train' : 0,
                                                                   'name' : '48000CIFAR10ResNet164',
                                                                   'depth' : 164
                                                                  }
                                                  },
                     '48000RESNET164_CIFAR10_TEST' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                   'train' : 1,
                                                                   'name' : '48000CIFAR10ResNet164',
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'Fast_resnet45000MixupCutout_train' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                   'train' : 0,
                                                                   'name' : 'Fast_resnet45000MixupCutout',
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'Fast_resnet45000MixupCutout_test' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                   'train' : 1,
                                                                   'name' : 'Fast_resnet45000MixupCutout',
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'Encoded_DR_train' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED_DR'],
                                                                   'train' : 0,
                                                                   'name' : 'Diabetic_Retinopathy',
                                                                   'fcount' : 7
                                                                  }
                                                  },
                     'Encoded_DR_test' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED_DR'],
                                                                   'train' : 1,
                                                                   'name' : 'Diabetic_Retinopathy',
                                                                   'fcount' : 3
                                                                  }
                                                  },
                     'Fast_resnet30000MixupCutout_train' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                   'train' : 0,
                                                                   'name' : 'Fast_resnet30000MixupCutout',
                                                                   'depth' : 164
                                                                  }
                                                  },
                     'Fast_resnet30000MixupCutout_test' : {'class_function' : 'encoded_dataset',
                                                  'init_params' : {'root' : DATASET_PATH['ENCODED256_CIFAR10'],
                                                                   'train' : 1,
                                                                   'name' : 'Fast_resnet30000MixupCutout',
                                                                   'depth' : 164
                                                                  }
                                                  }
                     }

