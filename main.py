import torch
import numpy as np
import matplotlib
from scipy import ndimage
import os, sys
import math
import pickle
import model_utils as modutil
import data_utils as datutil
import argparse
#import hmc

parser = argparse.ArgumentParser(description='train a feature extractor with GP / bayesian logistic reg / random feature expansion')
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, 
                    help="model setup <Kernel_name>+<GP/fixed> e.g. PreResNet+GP, encoded+GP")
parser.add_argument("--chkpt", type=str, default='',
                    help="location of model checkpoint to be loaded")
parser.add_argument("--fc", type=str, default='',
                    help="fully connected layer setup e.g. '100-10-5'")
parser.add_argument("--load", action='store_true',
                    help="whether model needs to be loaded from checkpoint")
parser.add_argument("--coef_load", nargs='*', default=[], 
                    help="checkpoint lists (decreasing priority) from which model coefficients will be partially initialized")
parser.add_argument("--datadir", type=str, default=None, help="data directory, default defined in data_config.py")
parser.add_argument("--train", type=str, help="train data")
parser.add_argument("--valid", type=str, help="validation data")
parser.add_argument("--test", type=str, help="test data")
parser.add_argument("--outsamp", type=str, help="out of sample detection data")
parser.add_argument("--train_size", type=int, help="train batch size")
parser.add_argument("--valid_size", type=int, help="validation batch size")
parser.add_argument("--test_size", type=int, help="test batch size")
parser.add_argument("--outsamp_size", type=int, help="out of sample data batch size")
parser.add_argument("--logfile", type=str, help="log file for the run")
parser.add_argument("--is_train", action='store_true', help="whether model will be trained or not")
parser.add_argument("--epoch", type=int, default=100, help="number of training epoch")
parser.add_argument("--numc", type=int, help="number of label class")
parser.add_argument("--wd", type=float, help="weight decay")
parser.add_argument("--stopAt", type=float, help="test accuracy at which training will be forced stop")
parser.add_argument("--depth", type=int, help="kernel depth e.g. for PreResNet110 depth = 110")
parser.add_argument("--gridsize", type=int, default=64, help="gridsize for gridinducing variational GP, e.g. 64")
parser.add_argument("--sgd", action='store_true', help="whether SGD is going to be used or not (ADAM)")
parser.add_argument("--deviceid", type=int, default=0, help="cuda gpu id")
parser.add_argument("--lr", type=float, nargs=2, help="learning rate start_value, end_value e.g. --lr 0.1 0.008")
parser.add_argument("--gp_ksize", type=int, help="output size of feature extractor of GP i.e. output size before applying fc layer")
args = parser.parse_args()

# Data loader initialization
trainloader1 = datutil.generate_dataloaders(args.train, batch_size=args.train_size, shuffle=False,
                                            num_workers=2, root=args.datadir)
testloader1 = datutil.generate_dataloaders(args.test, batch_size=args.test_size, shuffle=False,
                                           num_workers=2, root=args.datadir)
validloader1 = datutil.generate_dataloaders(args.outsamp, batch_size=args.outsamp_size, shuffle=False,
                                            num_workers=2, root=args.datadir)

# Cifar-100 interesting classes (used during out-of-class entropy calculation)
interesting_labels = [0, 1, 16, 17, 20, 21, 29, 39, 40, 49, 57, 71, 72, 73, 76]

# model to train/load/analyse
# user defined params
attribute_dict = {'model_type' : args.model, #"PreResNet+GP",     # <Kernel_name> + <GP>
    'saved_checkpoint_name' : args.chkpt, #"",
    'fc_setup' : list(map(int, args.fc.split('-'))),
    'load_model' : args.load,
    'partial_load' : args.coef_load,
    'program_outp_file_name' : args.logfile,
    'component_pretrained_mods' : args.coef_load,
    'train_model' : args.is_train,
    'train_epoch' : args.epoch,
    'num_classes' : args.numc,
    'weight_decay' : args.wd,
    'predef_test_acc' : args.stopAt,
    'depth' : args.depth,
    'grid_size' : args.gridsize,
    'optim_SGD' : args.sgd,
    'device' : torch.device('cuda:%d'%args.deviceid),
    'lr_init' : args.lr[0],
    'lr_final' : args.lr[1],
    'gp_kernel_feature' : args.coef_load, # 256, 640
    'print_init_model_state' : False}


if 'encoded' not in attribute_dict['model_type']:
    trainloader = trainloader1
    testloader = testloader1
    # validloader = validloader1
else:
    trainloader = trainloader2
    testloader = testloader2
    # validloader = validloader2

print('='*20, "Loading Model", '='*20,)
modutil.refresh_params()
for propt in attribute_dict:
    if propt in modutil.__dict__:
        modutil.__dict__[propt] = attribute_dict[propt]
    else:
        print("Model property '%s' not found!"%(propt))

# load / train the model
modutil.load_train(trainloader, testloader, attribute_dict['partial_load'])
if attribute_dict['train_model']:
    print("Saving model!")
    modutil.save_model(attribute_dict)

# param_list = param_chain if 'mcmc' in model['model_type'] else []
# Perform evaluation on model
# modutil.validate("out-of-class", validloader, interesting_labels=interesting_labels)
print("Validation (out of class) data analysis performed")

# modutil.validate("test", testloader)
print("Test data analysis performed")
