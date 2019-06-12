import torch
import numpy as np
from scipy import ndimage
import torch.optim as optim
import os, sys
import matplotlib.pyplot as plt
import math
import customized_model as model
import pickle
import datetime as dt
try:
    from gpytorch.likelihoods import SoftmaxLikelihood
    import gpytorch
except:
    print("gpytorch not installed, can not use GP features of this code")
# data path and path to saved models
DATA_PATH = '../data'
SAVED_MODEL_PATH = 'saved_models/'
SAVED_PRED_PATH = 'saved_predictions/'

# Number of samples to draw from weight priors to get marginal predictive likelihood
NUMBER_OF_SAMPLES_FOR_MARGINAL_PREDICTION = 10
program_outp_file_name = 'model_trajectory.txt'
output_writer = None #
saved_checkpoint_name = ""

load_model = True
train_model = False
perform_swa = False
stop_predef_acc = True
print_init_model_state = True

current_epoch = 0
train_epoch = 180                    # Retraining epoch when loading pretrained model to retrain
swa_epoch = 100                      # How many epochs for SWA params

model_type = None
kernel_net = None
num_classes = None
grid_size = 64
fc_setup = None
rendFeature_rank_reduction = None
gp_kernel_feature = None
predef_test_acc = None
lr_init = 0.1                        # Staring learning rate (0.1 if starting from scratch)
gp_layer_init = 0.001                # 
lr_final = 0.008                     # Final learning rate (also used for SWA)
momentum = 0.9                       # Momentum
weight_decay = None                  # Weight decay parameter
random_sample_train = 5              # Prior distribution sample size (used with bayesian net)
swa_update_freq = 1

# Module attributes holding important data which are training instance level
device = torch.device('cuda:1')
net = None
likelihood = None
optimizer = None
acc = 0
running_loss = 0
last_epoch = 0
last_lr = 0
depth = None
loss_mixing_ratio = 1.0


def push_output(string_to_write):
    "forward outputs to an output_file"
    
    global output_writer, program_outp_file_name
    
    if output_writer == None:
        print("Choosing progress output file ->", program_outp_file_name)
        output_writer = open(SAVED_MODEL_PATH + program_outp_file_name, 'w')
        
    output_writer.write(string_to_write)
    output_writer.flush()
    

def refresh_params():
    
    global fc_setup, current_epoch, lr_init, train_epoch, acc, lr_final, last_epoch

    fc_setup = None
    current_epoch = 0
    train_epoch = 180                    # Retraining epoch when loading pretrained model to retrain
    swa_epoch = 100                      # How many epochs for SWA params
    lr_init = 0.1                        # Staring learning rate (0.1 if starting from scratch)
    lr_final = 0.008                     # Final learning rate (also used for SWA)
    acc = 0
    last_epoch = 0
    rendFeature_rank_reduction = None
    depth = None
    output_writer = None


# Analysis function for train, test, out-of-class analysis
def validate(validation_type, dataloader, accuracy_only=False, interesting_labels=[], param_chain=[]):
    # validation type needs to be one of (train, test, out-of-class)
    
    global net, likelihood, acc, saved_checkpoint_name
    pred_list, target_list, entropy_list, prob_list = [[], [], [], []]
    
    dataiter = iter(dataloader)
    correct = 0
    total = 0
    token = ''
    loss = 0
    accuracy = 0
    brier_score = 0
    batch_count = 0
    net.eval()
    if '+GP' in model_type:
        likelihood.eval()
        
    if validation_type == "out-of-class" and len(interesting_labels) == 0:
        print("No interesting labels provided!! Rerun with labels as input")
        return

    with torch.no_grad():
        for data in dataiter:
            images, labels = data
            batch_count += 1
            if '+GP' not in net.net_type:
                outputs = net.avg_encode(images.to(device), NUMBER_OF_SAMPLES_FOR_MARGINAL_PREDICTION, param_chain)
            else:
                outputs = likelihood(net(images.to(device))).probs.mean(0)

            if len(labels.size()) == 1:
                scatter_lab = labels.unsqueeze(1)
            
            entropy = torch.sum(outputs * torch.log(outputs), dim=1)
            max_prob, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(device)).squeeze()

            if max(max_prob) > 1:
                print("softmax is wrong")
            
            if not accuracy_only:
                if validation_type != "out-of-class":

                    if '+GP' not in net.net_type:
                        batch_avg_loss = net.forward(images.to(device), labels.to(device), NUMBER_OF_SAMPLES_FOR_MARGINAL_PREDICTION)
                        loss = (loss * (batch_count - 1) + batch_avg_loss.item()) / (1.0 * batch_count)
                
                for prob, entrop, label, pred in zip(max_prob, entropy, labels, predicted):
                
                    pred_list.append(pred.item())
                    target_list.append(label.item())
                    prob_list.append(prob.item())
                    if (validation_type != 'out-of-class') or (label in interesting_labels):
                        entropy_list.append(-entrop.item())

            if validation_type != "out-of-class":
                correct += c.sum().item()
                total += labels.size(0)
    
    if validation_type != "out-of-class":
        acc = 100.0 * correct / total
        print("Accuracy statistics for :", validation_type)
        print('Overall accuracy : %2d %%' % (acc))
    
    _ = net.train()
    if '+GP' in model_type:
        _ = likelihood.train()

    if not accuracy_only:
        stat_dict = {'entrop' : entropy_list, 'predictions' : pred_list, 'targets' : target_list, 
                     'loss' : loss, 'probs' : prob_list}
        save_path_extension = 'validpred' if validation_type == 'out-of-class' else validation_type + 'pred'
        save_path = SAVED_PRED_PATH + saved_checkpoint_name + '.' + save_path_extension
        with open(save_path, 'wb') as predict_dict_file:
            pickle.dump(stat_dict, predict_dict_file)

    return

        
def load_train(trainloader, testloader):
    
    global net, likelihood, optimizer, depth, loss_mixing_ratio
    global current_epoch, lr_init, train_epoch, print_init_model_state
    global acc, running_loss, last_epoch, last_lr
    
    running_loss = 0.0
    
    if '+GP' not in model_type:
        net = model.__dict__[kernel_net](device=device, num_classes=num_classes, depth=depth,
                                   rendFeature_rank_reduction=rendFeature_rank_reduction, 
                                   loss_mixing_ratio=loss_mixing_ratio, net_type=model_type, 
                                   fc_setup=fc_setup, trainloader=trainloader)
        net.to(device)
        optimizer = optim.SGD(net.parameters(), lr=lr_init, weight_decay=weight_decay, momentum=momentum)
        _ = net.train()
        likelihood = None
    else:
        net = model.GPNet(device=device, kernel_net=kernel_net, kernel_net_type=model_type,
                          gp_feature_size=gp_kernel_feature, num_classes=num_classes, depth=depth, grid_size=grid_size)
        net.to(device)
        likelihood = SoftmaxLikelihood(gp_kernel_feature, num_classes)
        likelihood.to(device)
        optimizer = optim.SGD([
        {'params': net.feature_extractor.parameters()},
        {'params': net.gp_layer.hyperparameters(), 'lr': lr_init * 0.01},
        {'params': net.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
                        ], lr=lr_init, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        _ = net.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, net.gp_layer, num_data=len(trainloader.dataset))

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("total number of parameters is", pytorch_total_params)

    # load model from disk
    if load_model:
        checkpoint = torch.load(SAVED_MODEL_PATH + saved_checkpoint_name + '.chkpt')
        print("Model state loaded")
        net.load_state_dict(checkpoint['model_state'])

        if 'randFeature' in net.net_type:
            net.rand_W = checkpoint['rand_W'].to(device)
            net.rand_B = checkpoint['rand_B'].to(device)
        
        if '+GP' in net.net_type:
            likelihood.load_state_dict(checkpoint['likelihood_state'])
            print("Likelihood state loaded")
            
        print("Model is loaded! Loss = %.3f and accuracy = %.3f %%" %(checkpoint['loss'], checkpoint['acc'] \
                                                      if checkpoint['acc'] > 1 else 100*checkpoint['acc']))
        net.train()
        optimizer.load_state_dict(checkpoint['optim_state'])
        print("Optimizer is loaded!")
        current_epoch = checkpoint['epoch']
        lr_init = checkpoint['last_lr']
        print("Current lr is: %.3f and target lr is: %.3f" %(lr_init, lr_final))

    # derived params
    epoch_count = current_epoch
    epoch_count += train_epoch * int(train_model)
    swa_start = epoch_count
    epoch_count += swa_epoch * int(perform_swa)
    end_epoch = swa_start if 'SWA' in model_type else epoch_count

    if print_init_model_state:
        for name, param in net.named_parameters():
            print(name, param.size(), torch.max(param.data), torch.min(param.data))

    # perform train/swa update
    # with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
    for epoch in range(current_epoch, epoch_count):  # loop over the dataset multiple times

            factor = model.learning_rate_mod_factor(model_type, epoch, end_epoch, lr_init, lr_final, running_loss)
            for i, g in enumerate(optimizer.param_groups):
                print("Learning rate for param %d is currently %.4f" %(i, g['lr']))
                push_output("Learning rate for param %d is currently %.4f\n" %(i, g['lr']))
                g['lr'] = lr_init * factor
#                 if i == 1 and '+GP' in model_type:
#                     g['lr'] = lr_init * factor * 0.01
                print("Learning rate for param %d has been changed to %.4f" %(i, g['lr']))
                push_output("Learning rate for param %d has been changed to %.4f\n" %(i, g['lr']))

            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                if '+GP' not in model_type:
                    loss = net.forward(inputs, labels, random_sample_train)
                else:
                    output = net(inputs)
                    loss = -mll(output, labels)
                loss.backward()
                #net.modify_grad()
                optimizer.step()
                running_loss = 0.9*running_loss + 0.1*loss.item() if running_loss != 0 else loss.item()
                if i%500 == 0:
                    print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                    push_output('[%d, %5d] loss: %.3f\n' %(epoch + 1, i + 1, running_loss))

            last_lr = lr_init * factor
            last_epoch = epoch
            print("=== Accuracy using SGD params ===")
            push_output("=== Accuracy using SGD params ===\n")
            validate("test", testloader, accuracy_only=True)
            push_output('Overall accuracy : %2d %%\n' % (acc))
            if stop_predef_acc:
                if acc >= predef_test_acc and epoch >= current_epoch + 0.7*(epoch_count - current_epoch):
                    print("Stopped because accuracy reached")
                    push_output("Stopped because accuracy reached\n")
                    break
    
    print('Model is ready')
    

def save_model():
    
    # Save model, optim and some stats
    attributes = [('depth', depth), ('lr', lr_init), ('mom', momentum), ('wd', weight_decay),
                  ('FC', '-'.join(map(str, fc_setup))), ('acc', acc)]
    
    checkpoint = {'epoch' : last_epoch,
                  'model_state' : net.state_dict(),
                  'optim_state' : optimizer.state_dict(),
                  'loss' : running_loss,
                  'acc' : acc,
                  'last_lr' : last_lr}
    
    if 'randFeature' in net.net_type:
        checkpoint['rand_W'] = net.rand_W
        checkpoint['rand_B'] = net.rand_B
        attributes = [('randomFeatureRank', rendFeature_rank_reduction)] + attributes
        
    if '+GP' in net.net_type or '+GP' in model_type:
        checkpoint['likelihood_state'] = likelihood.state_dict()
        
    checkpoint_name = model_type + '_' + '_'.join(['-'.join(map(str, item)) for item in attributes])
    curtime = dt.datetime.now()
    tm = curtime.strftime("%Y-%m-%d-%H.%M")    
    torch.save(checkpoint, SAVED_MODEL_PATH + checkpoint_name + '-' + tm + '.chkpt')
    print("Model and optimizer status has been saved!")
    

def return_model():
    
    return net

