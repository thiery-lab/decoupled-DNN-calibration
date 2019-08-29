import torch
import numpy as np
from scipy import ndimage
import torch.optim as optim
import os, sys
import matplotlib.pyplot as plt
import math
from models import *
import pickle
import datetime as dt
from tqdm import tqdm
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
output_writer = None
saved_checkpoint_name = ""
component_pretrained_mods = []

load_model = True
train_model = False
perform_swa = False
stop_predef_acc = True
print_init_model_state = True
ece_mid_list, ece_avg_list = [], []

save_freq = 100
current_epoch = 0
train_epoch = 180                    # Retraining epoch when loading pretrained model to retrain
swa_epoch = 100                      # How many epochs for SWA params
end_epoch = None

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
gp_weight_decay = None               # Weight decay specific to GP layer
random_sample_train = 5              # Prior distribution sample size (used with bayesian net)
swa_update_freq = 1

# Module attributes holding important data which are training instance level
device = None
ngpu = 0
net = None
likelihood = None
optimizer = None
optim_SGD = True
acc = 0
running_loss = 0
last_epoch = 0
last_lr = 0
depth = None
loss_mixing_ratio = 1.0

lr_decay_start_t = 0.4
loss_direction = []
last_loss = -100
accuracy = 0.1


def learning_rate_mod_factor(epoch, loss):
    
    global lr_decay_start_t, loss_direction, last_loss, net_type, end_epoch, lr_init, lr_final
        
    t = (epoch) / (end_epoch*1.0)
    
    if t < 0.4 and lr_decay_start_t == 0.4:
        if last_loss != -100:
            loss_direction.append(last_loss < loss)
            if len(loss_direction) > 10:
                loss_direction = loss_direction[1:]
                print("Loss increment instances so far = %d" %(sum(loss_direction)))
                if sum(loss_direction) > 4:
                    lr_decay_start_t = t - 0.00001

    last_loss = loss
    lr_ratio = lr_final / lr_init
    if t <= lr_decay_start_t:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - lr_decay_start_t) / (0.9 - lr_decay_start_t)
    else:
        factor = lr_ratio
    
    return factor


def push_output(string_to_write):
    "forward outputs to an output_file"
    
    global output_writer, program_outp_file_name
    
    if output_writer == None:
        print("Choosing progress output file ->", program_outp_file_name)
        output_writer = open(SAVED_MODEL_PATH + program_outp_file_name, 'w')
        
    output_writer.write(string_to_write)
    output_writer.flush()
    

def refresh_params():
    
    global fc_setup, current_epoch, lr_init, train_epoch, acc, lr_final, last_epoch, optim_method

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
    optim_method = 'SGD'


# Analysis function for train, test, out-of-class analysis
def validate(validation_type, dataloader, accuracy_only=False, interesting_labels=[], param_chain=[]):
    # validation type needs to be one of (train, test, out-of-class)
    
    global net, likelihood, acc, saved_checkpoint_name, ece_avg_list, ece_mid_list
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
        for data in tqdm(dataiter, ncols=100):
            images, labels = data
            batch_count += 1
            if '+GP' not in net.net_type:
                outputs = net.avg_encode(images.to(device), NUMBER_OF_SAMPLES_FOR_MARGINAL_PREDICTION, param_chain)
            else:
                outp = likelihood(net(images.to(device))).probs
                outputs = outp.mean(0)

                if validation_type != "out-of-class":
                    for index in range(outp.shape[0]):
                        probs = outp[index]
                        current_loss = torch.log(torch.gather(probs, dim=1, index=labels.to(device).reshape((len(labels), 1))))
                        current_loss = -torch.sum(current_loss) / outp.shape[0]

                    loss = (loss * (batch_count - 1) + current_loss) / (1.0 * batch_count)

            entropy = torch.sum(outputs * torch.log(outputs), dim=1)
            max_prob, predicted = torch.max(outputs, 1)
            c = (predicted == labels.to(device)).squeeze()

            if max(max_prob) > 1:
                print("softmax is wrong")
            
            if not accuracy_only:
                if validation_type != "out-of-class":

                    if '+GP' not in net.net_type:
                        batch_avg_loss = net.forward(images.to(device), labels.to(device), NUMBER_OF_SAMPLES_FOR_MARGINAL_PREDICTION).sum()
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

    bins = [(p + 1) / 30.0 for p in range(30)]
    ece_mid, ece_avg = calculate_ECE(probs=prob_list, preds=pred_list, targets=target_list, ECE_bin=bins)
    ece_mid_list.append(ece_mid)
    ece_avg_list.append(ece_avg)
    print('ECE values are %.3f, %.3f when mid bin and avg used respectively' % (ece_mid, ece_avg))

    if not accuracy_only:
        stat_dict = {'entrop': entropy_list, 'predictions': pred_list, 'targets': target_list,
                     'loss': loss, 'probs': prob_list, 'ece': (ece_mid, ece_avg),
                     'ece_mid_traj': ece_mid_list, 'ece_avg_traj': ece_avg_list}
        save_path_extension = 'validpred' if validation_type == 'out-of-class' else validation_type + 'pred'
        save_path = SAVED_PRED_PATH + saved_checkpoint_name + '.' + save_path_extension
        with open(save_path, 'wb') as predict_dict_file:
            pickle.dump(stat_dict, predict_dict_file)

    return

        
def load_train(trainloader, testloader):
    
    global net, likelihood, optimizer, depth, loss_mixing_ratio, gp_kernel_feature
    global current_epoch, lr_init, train_epoch, print_init_model_state, save_freq
    global acc, running_loss, last_epoch, last_lr, end_epoch, optim_SGD, ngpu, gp_weight_decay
    
    running_loss = 0.0
    
    if '+GP' not in model_type:
        net = BayesFCNet(device=device, num_classes=num_classes, depth=depth,
                        rendFeature_rank_reduction=rendFeature_rank_reduction, 
                        loss_mixing_ratio=loss_mixing_ratio, net_type=model_type,
                        fc_setup=fc_setup, trainloader=trainloader, feature_size=gp_kernel_feature)
        net.to(device)
        if optim_SGD:
            optimizer = optim.SGD(net.parameters(), lr=lr_init, weight_decay=weight_decay, momentum=momentum)
        else:
            optimizer = optim.SGD(net.parameters(), lr=lr_init, weight_decay=weight_decay)
        _ = net.train()
        likelihood = None
    else:
        net = GPNet(device=device, kernel_net_type=model_type, gp_feature_size=gp_kernel_feature, 
                    num_classes=num_classes, depth=depth, grid_size=grid_size)
        net.to(device)
        likelihood = SoftmaxLikelihood(gp_kernel_feature, num_classes)
        likelihood.to(device)
        if optim.SGD:
            optimizer = optim.SGD([
            {'params': net.feature_extractor.parameters(), 'weight_decay': weight_decay},
            {'params': net.gp_layer.hyperparameters(), 'lr': lr_init * 0.01, 'weight_decay': gp_weight_decay},
            {'params': net.gp_layer.variational_parameters(), 'weight_decay': gp_weight_decay},
            {'params': likelihood.parameters()},
                            ], lr=lr_init, momentum=momentum, nesterov=True)  # , weight_decay=weight_decay)
        else:
            optimizer = optim.Adam([
            {'params': net.feature_extractor.parameters(), 'weight_decay': weight_decay},
            {'params': net.gp_layer.hyperparameters(), 'lr': lr_init * 0.01, 'weight_decay': gp_weight_decay},
            {'params': net.gp_layer.variational_parameters(), 'weight_decay': gp_weight_decay},
            {'params': likelihood.parameters(), 'weight_decay': gp_weight_decay},
                            ], lr=lr_init)  # , weight_decay=weight_decay)
        _ = net.train()
        likelihood.train()
        mll = gpytorch.mlls.VariationalELBO(likelihood, net.gp_layer, num_data=len(trainloader.dataset))

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print("total number of parameters is", pytorch_total_params)

    # load model from disk
    if load_model:
        if os.path.exists(SAVED_MODEL_PATH + saved_checkpoint_name + '.chkpt'):
            checkpoint = torch.load(SAVED_MODEL_PATH + saved_checkpoint_name + '.chkpt', map_location=device)
        elif os.path.exists(SAVED_MODEL_PATH + saved_checkpoint_name + '.interim'):
            checkpoint = torch.load(SAVED_MODEL_PATH + saved_checkpoint_name + '.interim', map_location=device)
        else:
            print("Neither checkpoint nor iterim file found! check file name")
            sys.exit(-1)

        print("Model state loaded")
        if 'command_dict' in checkpoint:
            print('Command directory was as follows\n',
                  checkpoint['command_dict'].__repr__().replace(', ', ',\n'))

        current_state = net.state_dict()
        state_dict_to_load = checkpoint['model_state']
        current_state.update(state_dict_to_load)
        net.load_state_dict(current_state)

        if 'randFeature' in net.net_type:
            net.rand_W = checkpoint['rand_W'].to(device)
            net.rand_B = checkpoint['rand_B'].to(device)

        if '+GP' in net.net_type:
            likelihood.load_state_dict(checkpoint['likelihood_state'])
            print("Likelihood state loaded")

        print("Model is loaded! Loss = %.3f and accuracy = %.3f %%" %(checkpoint['loss'], checkpoint['acc'] \
                                                      if checkpoint['acc'] > 1 else 100*checkpoint['acc']))
        optimizer.load_state_dict(checkpoint['optim_state'])
        print("Optimizer is loaded!")
        current_epoch = checkpoint['epoch']
        lr_init = checkpoint['last_lr']
        print("Current lr is: %.3f and target lr is: %.3f" %(lr_init, lr_final))

    elif len(component_pretrained_mods) > 0:

        current_state = net.state_dict()
        state_dict_to_load = dict()

        for pretrained_mod in component_pretrained_mods:
            if os.path.exists(SAVED_MODEL_PATH + pretrained_mod + '.chkpt'):
                checkpoint = torch.load(SAVED_MODEL_PATH + pretrained_mod + '.chkpt', map_location=device)
            elif os.path.exists(SAVED_MODEL_PATH + pretrained_mod + '.interim'):
                checkpoint = torch.load(SAVED_MODEL_PATH + pretrained_mod + '.interim', map_location=device)
            else:
                print("Neither checkpoint nor iterim file found! check file name")
                sys.exit(-1)
            checkpoint_dict = checkpoint['model_state']
            for k, v in checkpoint_dict.items():
                if k in current_state:
                    state_dict_to_load[k] = v
                elif 'feature_extractor.' + k in current_state:
                    state_dict_to_load['feature_extractor.' + k] = v
            
            if '+GP' in net.net_type and 'likelihood_state' in checkpoint:
                likelihood.load_state_dict(checkpoint['likelihood_state'])
                print("Likelihood state loaded")

        current_state.update(state_dict_to_load)
        print("Partial loading will update only %d parameters out of %d parameters" % (len(state_dict_to_load),
                                                                                           len(current_state)))
        net.load_state_dict(current_state)
            
    _ = net.train()

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

        if optim_SGD:
            factor = learning_rate_mod_factor(epoch, running_loss)
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
            loss.sum().backward()
            # net.modify_grad()
            optimizer.step()
            running_loss = 0.9*running_loss + 0.1*loss.item() if running_loss != 0 else loss.item()
            if i% (len(trainloader) // 2) == 0:
                print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss))
                push_output('[%d, %5d] loss: %.3f\n' %(epoch + 1, i + 1, running_loss))

        last_lr = lr_init * factor if optim_SGD else lr_init
        last_epoch = epoch
        print("=== Accuracy using SGD params ===")
        push_output("=== Accuracy using SGD params ===\n")
        validate("test", testloader, accuracy_only=True)
        push_output('Overall accuracy : %2d %%\n' % (acc))
        if epoch % save_freq == 0 and epoch != 0:
            save_model(None, True)
        if stop_predef_acc:
            if acc >= predef_test_acc and epoch >= current_epoch + 0.7*(epoch_count - current_epoch):
                print("Stopped because accuracy reached")
                push_output("Stopped because accuracy reached\n")
                break

    print('Model is ready')
    

def save_model(cmd_dict, interim=False):

    global net, saved_checkpoint_name
    # Save model, optim and some stats
    attributes = [('depth', depth), ('lr', lr_init), ('mom', momentum), ('wd', weight_decay),
                  ('gpwd', gp_weight_decay), ('FC', '-'.join(map(str, fc_setup))), ('acc', '%.2f'%acc)]
    
    checkpoint = {'epoch': last_epoch,
                  'model_state': net.state_dict(),
                  'optim_state': optimizer.state_dict(),
                  'loss': running_loss,
                  'acc': acc,
                  'last_lr': last_lr,
                  'command_dict' : cmd_dict}
    
    if 'randFeature' in net.net_type:
        checkpoint['rand_W'] = net.rand_W
        checkpoint['rand_B'] = net.rand_B
        attributes = [('randomFeatureRank', rendFeature_rank_reduction)] + attributes
        
    if '+GP' in net.net_type or '+GP' in model_type:
        checkpoint['likelihood_state'] = likelihood.state_dict()
        
    checkpoint_name = model_type + '_' + '_'.join(['-'.join(map(str, item)) for item in attributes])
    curtime = dt.datetime.now()
    tm = curtime.strftime("%Y-%m-%d-%H.%M")
    if not interim:
        final_name = SAVED_MODEL_PATH + checkpoint_name + '-' + tm
        saved_checkpoint_name = checkpoint_name + '-' + tm
        final_name += '.chkpt'
    else:
        final_name = SAVED_MODEL_PATH + checkpoint_name + '.interim'
    torch.save(checkpoint, final_name)
    print("Model and optimizer status has been saved!")
    

def return_model():
    
    return net


def calculate_ECE(probs, preds, targets, ECE_bin=None):
    
    if ECE_bin is None:
        ECE_bin = np.sort(probs)[::len(probs)//20].tolist()
        
    if ECE_bin[0] == 0:
        ECE_bin = ECE_bin[1:]
    if ECE_bin[-1] != 1:
        ECE_bin += [1]
    ECE_bin_correct = [0 for _ in range(len(ECE_bin))]
    ECE_bin_total = [0 for _ in range(len(ECE_bin))]
    ECE_bin_total_conf = [0 for _ in range(len(ECE_bin))]

    for index in range(len(probs)):
        for bin_ in range(len(ECE_bin)):
            if probs[index] <= ECE_bin[bin_]:
                ECE_bin_correct[bin_] += int(targets[index] == preds[index])
                ECE_bin_total[bin_] += 1
                ECE_bin_total_conf[bin_] += probs[index]
                break

    ece_score_mid = 0
    ece_score_avg = 0
    start_bin = [0] + ECE_bin[:-1]
    mid_bins = [0.5*(start_bin[i] + ECE_bin[i]) for i in range(len(ECE_bin))]
    
#     print(mid_bins, ECE_bin_correct, ECE_bin_total)
    for prob_class in range(len(ECE_bin)):
        correct = ECE_bin_correct[prob_class]
        total = ECE_bin_total[prob_class]
        avg_conf = ECE_bin_total_conf[prob_class]*1.0 / total if total > 0 else 0
        accuracy = float(correct)/total if total > 0 else 0
        ece_score_mid += abs(accuracy - mid_bins[prob_class]) * total
        ece_score_avg += abs(accuracy - avg_conf) * total
        
    ece_score_mid /= 1.0*sum(ECE_bin_total)
    ece_score_avg /= 1.0*sum(ECE_bin_total)
    return ece_score_mid, ece_score_avg


def encode_dump(file_name, dataloader, evalmode=False):
    """
    function to encode test, train, validation data by
    feature extractor of the model and store the encoded
    data in the desired directory
    
    file_name : name format of file series eg: 'encoded28x10WideResNet_CIFAR10_640_valid'
                 including directory if saving to a folder
    dataloader : train/test/valid loader for encoding
    evalmode : true if model needs to be switched to eval() before encoding
    """
    
    global device, net
    
    if evalmode:
        net.eval()
    import json
    data = {'feature': np.array([]), 'label': np.array([])}
    file_count = 0

    for i, dat in enumerate(dataloader, 0):

        inputs, labels = dat
        inputs = inputs.to(device)
        labels = labels.to(device)
        if '+GP' in model_type:
            output = net.feature_extractor.extract_feature(inputs)
        else:
            output = net.extract_feature(inputs)
        
        if len(data['feature']) > 0:
            data['feature'] = np.concatenate((data['feature'], output.detach().cpu().numpy()), axis=0)
            data['label'] = np.concatenate((data['label'], labels.detach().cpu().numpy()), axis=0)
        else:
            data['feature'] = output.detach().cpu().numpy()
            data['label'] = labels.detach().cpu().numpy()

        if len(data['feature'])//10000 > 0:

            data['feature'] = data['feature'].tolist()
            data['label'] = data['label'].tolist()

            file_count += 1
            file_n = file_name + str(file_count)
            print("dumping", len(data['label']), "size data at", file_n)
            with open(file_n, 'wb') as part_pickle:
                pickle.dump(data, part_pickle)
            data = {'feature': np.array([]), 'label': np.array([])}

    if len(data['feature']) > 0:

        data['feature'] = data['feature'].tolist()
        data['label'] = data['label'].tolist()

        file_count += 1
        file_n = file_name + str(file_count)
        print("dumping", len(data['label']), "size data at", file_n)
        with open(file_n, 'wb') as part_pickle:
            pickle.dump(data, part_pickle)
        data = {'feature': np.array([]), 'label': np.array([])}

    if evalmode:
        _ = net.train()
    

