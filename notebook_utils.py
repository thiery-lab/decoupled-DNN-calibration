import torch
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import os, sys
import math
import pickle
import data_utils as datutil
import datetime as dt
import hmc
from models import *
import gpytorch
import torch.nn.functional as F


def validate(model, **kwargs):
    
    dataloader = kwargs['dataloader']
    likelihood = kwargs.get('likelihood', None)
    device = kwargs.get('device', None)
    savefile = kwargs.get('savefile', None)
    num_sample = kwargs.get('num_sample', 30)
    params = kwargs.get('params', None)
    weights = kwargs.get('weights', None)
    if weights is not None:
        if 'torch' in weights.type():
            weights = weights.type(torch.FloatTensor)
        else:
            weights = torch.tensor(weights).type(torch.FloatTensor)

    dataiter = iter(dataloader)
    correct, total, loss_preavg, loss_postavg, accuracy, brier_score, batch_count = 0, 0, 0, 0, 0, 0, 0
    w_correct, w_total = 0, 0
    pred_list, target_list, prob_list, all_prob_list = [], [], [], []
    model.eval()
    if likelihood is not None:
        likelihood.eval()
        
    with torch.no_grad():
        for data in dataiter:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            batch_count += 1
            
            # `outp` is expected to contain (num_sample x batch_size x num_classes)
            # i.e. output probabilities for different parameters sampled from posterior
            if likelihood is not None:
                with gpytorch.settings.num_likelihood_samples(30):
                    outp = likelihood(model(images)).probs
            elif params is None:
                try:
                    outp = model.infer(images, num_sample=num_sample)
                except:
                    outp = model(images)
                    outp = F.softmax(outp, dim=1)
                    outp = torch.unsqueeze(outp, dim=0)
            else:
                outp = model.infer(images, samples=params)
            
            # outputs is calculated by averaging over probabilities over posterior samples
            outputs = outp.mean(0)

            # pre-averaging loss is calculated by calculating loss first and then averaging
            current_loss = 0
            for index in range(outp.shape[0]):
                probs = outp[index]
                temp_loss = torch.log(torch.gather(probs, dim=1, index=labels.reshape((len(labels), 1))))
                current_loss -= torch.sum(temp_loss) / (outp.shape[1] * outp.shape[0])
            loss_preavg = (loss_preavg * (batch_count - 1) + current_loss) / (1.0 * batch_count)
            
            # post-averaging loss is calculated by using the marginal class probabilities
            current_loss = 0
            temp_loss = torch.log(torch.gather(outputs, dim=1, index=labels.reshape((len(labels), 1))))
            current_loss = -torch.sum(temp_loss) / outputs.shape[0]
            loss_postavg = (loss_postavg * (batch_count - 1) + current_loss) / (1.0 * batch_count)

            # entropy, max class probability
            entropy = torch.sum(outputs * torch.log(outputs), dim=1)
            max_prob, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            correct += c.sum().item()
            total += labels.size(0)

            if weights is not None:
                w_c = c.type(torch.FloatTensor) * weights[labels]
                w_correct += w_c.sum().item()
                w_total += weights[labels].sum().item()
            
            # lists are populated for calculating ECE
            for prob, entrop, label, pred, all_probs in zip(max_prob, entropy, labels, predicted, outputs):
                pred_list.append(pred.item())
                target_list.append(label.item())
                prob_list.append(prob.item())
                all_prob_list.append(all_probs)
    
        acc = 100.0 * correct / total
        print("Accuracy statistics")
        print('Overall accuracy : %.1f %%' %acc)
        if weights is not None:
            w_acc = 100.0 * w_correct / w_total
            print('Weight adjusted accuracy : %1f %%' %w_acc)

    _ = model.train()
    if likelihood is not None:
        _ = likelihood.train()

    bins = [(p + 1) / 30.0 for p in range(30)]
    ece_mid, ece_avg, sce_score = calculate_ECE(all_probs=all_prob_list, targets=target_list,
                                                probs=prob_list, preds=pred_list, ECE_bin=bins)
    
    print('ECE values are %.3f, %.3f when mid bin and avg used respectively' % (ece_mid, ece_avg))
    print('SCE values are %.5f' % sce_score)
    print('Pre-averaging loss:', loss_preavg, 'Post-averaging loss:', loss_postavg)

    # stats regarding predictions are saved in a file
    if savefile is not None:
        stat_dict = {'entrop': entropy_list, 'predictions': pred_list, 'targets': target_list,
                     'loss_preavg': loss_preavg, 'loss_postavg': loss_postavg, 'probs': prob_list, 'sce': sce_score,
                     'ece': (ece_mid, ece_avg), 'ece_mid_traj': ece_mid_list, 'ece_avg_traj': ece_avg_list}
        save_path_extension = 'validpred' if validation_type == 'out-of-class' else validation_type + 'pred'
        save_path = 'saved_predictions/' + savefile + '.' + save_path_extension
        with open(save_path, 'wb') as predict_dict_file:
            pickle.dump(stat_dict, predict_dict_file)

    return acc, ece_avg, sce_score


def calculate_ECE(all_probs, targets, probs=None, preds=None, ECE_bin=None):
    '''
    Args:
        all_probs: probability matrix of size (data_size x num_class) containing
            all the class probabilities for all observations
        targets: actual class labels for all observations
        probs: max class probability for all observations, if not provided, will 
            be calculated from all_probs
        preds: predictions for all observations, if not provided, will be calculated
            from all_probs
        ECE_bin: bins for calculating ECE / SCE, if not provided then auto-binning is
            done based on max probability quantiles
        
    Returns:
        ECE_mid: ECE score when at each bin mid bin is used as confidence
        ECE_avg: ECE score when at each bin weighted average of probabilities
            are used as confidence
        SCE_score: SCE score with confidence as weighted average of probabilities
            refer to 'https://arxiv.org/pdf/1904.01685.pdf'
    '''
    all_probs = np.array(all_probs)
    if probs is None:
        probs = np.max(all_probs, 1)
    if preds is None:
        preds = np.argmax(all_probs, 1)
    num_classes = len(all_probs[0])
        
    if ECE_bin is None:
        ECE_bin = np.sort(probs)[::len(probs)//20].tolist()
        
    if ECE_bin[0] == 0:
        ECE_bin = ECE_bin[1:]
    if ECE_bin[-1] != 1:
        ECE_bin += [1]
    ECE_bin_correct = [0 for _ in range(len(ECE_bin))]
    ECE_bin_total = [0 for _ in range(len(ECE_bin))]
    ECE_bin_total_conf = [0 for _ in range(len(ECE_bin))]
    SCE_bin_correct = [[0 for _ in range(len(ECE_bin))] for class_ in range(num_classes)]
    SCE_bin_total = [[0 for _ in range(len(ECE_bin))] for class_ in range(num_classes)]
    SCE_bin_total_conf = [[0 for _ in range(len(ECE_bin))] for class_ in range(num_classes)]

    for index in range(len(probs)):
        for bin_ in range(len(ECE_bin)):
            if probs[index] <= ECE_bin[bin_]:
                ECE_bin_correct[bin_] += int(targets[index] == preds[index])
                ECE_bin_total[bin_] += 1
                ECE_bin_total_conf[bin_] += probs[index]
                break
        for class_ in range(num_classes):
            for bin_ in range(len(ECE_bin)):
                if all_probs[index][class_] <= ECE_bin[bin_]:
                    SCE_bin_correct[class_][bin_] += int(class_ == preds[index])
                    SCE_bin_total[class_][bin_] += 1
                    SCE_bin_total_conf[class_][bin_] += all_probs[index][class_]
                    break

    ece_score_mid = 0
    ece_score_avg = 0
    sce_score = 0
    start_bin = [0] + ECE_bin[:-1]
    mid_bins = [0.5*(start_bin[i] + ECE_bin[i]) for i in range(len(ECE_bin))]
    
    for prob_class in range(len(ECE_bin)):
        correct = ECE_bin_correct[prob_class]
        total = ECE_bin_total[prob_class]
        avg_conf = ECE_bin_total_conf[prob_class]*1.0 / total if total > 0 else 0
        accuracy = float(correct)/total if total > 0 else 0
        ece_score_mid += abs(accuracy - mid_bins[prob_class]) * total
        ece_score_avg += abs(accuracy - avg_conf) * total
        
    for class_ in range(num_classes):
        sce_score_class = 0
        for prob_class in range(len(ECE_bin)):
            correct = SCE_bin_correct[class_][prob_class]
            total = SCE_bin_total[class_][prob_class]
            avg_conf = SCE_bin_total_conf[class_][prob_class]*1.0 / total if total > 0 else 0
            accuracy = float(correct)/total if total > 0 else 0
            sce_score_class += abs(accuracy - avg_conf) * total

        sce_score += sce_score_class / (len(all_probs)*num_classes)
        
    ece_score_mid /= 1.0*sum(ECE_bin_total)
    ece_score_avg /= 1.0*sum(ECE_bin_total)
    return ece_score_mid, ece_score_avg, sce_score


def encode_dump(net, file_name, dataloader, device, evalmode=False):
    """
    function to encode test, train, validation data by
    feature extractor of the model and store the encoded
    data in the desired directory
    
    file_name : name format of file series eg: 'encoded28x10WideResNet_CIFAR10_640_valid'
                 including directory if saving to a folder
    dataloader : train/test/valid loader for encoding
    evalmode : true if model needs to be switched to eval() before encoding
    """
    
    if evalmode:
        net.eval()
    import json
    data = {'feature': np.array([]), 'label': np.array([])}
    file_count = 0

    for i, dat in enumerate(dataloader, 0):

        inputs, labels = dat
        inputs = inputs.to(device)
        labels = labels.to(device)
        output = net(inputs)
        
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
    
    
class limiting_ECE_loss(nn.Module):
    def __init__(self, reduction='avg'):
        """
            
        """
        super(limiting_ECE_loss, self).__init__()
        self.reduction = reduction

    def forward(self, outputs, labels):
        """
        """
        probs = F.softmax(outputs, dim=1)
        max_prob, predicted = torch.max(probs, 1)
        c = (predicted == labels.to(self.device)).squeeze().type(torch.FloatTensor)
        batch_size = outputs.size()[0]
        loss = torch.sum(torch.abs(max_prob - c.to(self.device)))
        if self.reduction == 'avg':
            loss /= batch_size
        return loss
    
    def to(self, device):
        self.device = device
        

class limiting_SCE_loss(nn.Module):
    def __init__(self, reduction='avg'):
        """
            
        """
        super(limiting_SCE_loss, self).__init__()
        self.reduction = reduction
        
    def forward(self, outputs, labels):
        """
        """
        probs = F.softmax(outputs, dim=1)
        batch_size, num_class = probs.size()
        one_hot = torch.zeros(outputs.size(), device=self.device)
        torch.scatter_(one_hot, 1, labels.reshape(batch_size, 1), 1)
        loss = torch.sum(torch.abs(probs - one_hot)) / (batch_size * num_class)
        if self.reduction == 'avg':
            loss /= batch_size
        return loss

    def to(self, device):
        self.device = device