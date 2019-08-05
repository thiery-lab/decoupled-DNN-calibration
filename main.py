import torch
import numpy as np
import matplotlib
from scipy import ndimage
import os, sys
import math
import pickle
import model_utils as modutil
import data_utils as datutil
import hmc

# Data loader initialization
trainloader1 = datutil.generate_dataloaders('DIAB_RETIN_TRAIN', batch_size=7, shuffle=True, num_workers=2)
testloader1 = datutil.generate_dataloaders('DIAB_RETIN_TEST', batch_size=5, shuffle=False, num_workers=2)

# Cifar-100 interesting classes (used during out-of-class entropy calculation)
interesting_labels = [0, 1, 16, 17, 20, 21, 29, 39, 40, 49, 57, 71, 72, 73, 76]

# model to train/load/analyse
# user defined params
models = [{'model_type' : "PreResNet110",     # <Kernel_name> + <GP>
    'saved_checkpoint_name' : "",
    'fc_setup' : [2],
    'load_model' : False,
    'train_model' : True,
    'train_epoch' : 50,
    'num_classes' : 2,
    'weight_decay' : 3e-4,
    'predef_test_acc' : 90,
    'depth' : 110,
    'grid_size' : 64,
    'lr_init' : 0.01,
    'lr_final' : 0.001,
    'optim_SGD' : False,
    'device' : torch.device('cuda:1'),
    'gp_kernel_feature' : 256, # 256, 640
    'print_init_model_state' : False},]

for model in models:
    
    if 'encoded' not in model['model_type']:
        trainloader = trainloader1
        testloader = testloader1
        # validloader = validloader1
    else:
        trainloader = trainloader2
        testloader = testloader2
        # validloader = validloader2
        
    print('='*20, "Loading Model", '='*20,)
    modutil.refresh_params()
    for propt in model:
        if propt in modutil.__dict__:
            modutil.__dict__[propt] = model[propt]
        else:
            print("Model property '%s' not found!"%(propt))
    
    #load / train the model
    modutil.load_train(trainloader, testloader)
    if model['train_model']:
        print("Saving model!")
        modutil.save_model()

    # param_list = param_chain if 'mcmc' in model['model_type'] else []
    # Perform evaluation on model
    # modutil.validate("out-of-class", validloader, interesting_labels=interesting_labels)
    print("Validation (out of class) data analysis performed")

    #modutil.validate("test", testloader)
    print("Test data analysis performed")
