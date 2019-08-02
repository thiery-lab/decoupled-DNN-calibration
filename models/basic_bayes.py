import torch
import numpy as np
import os, sys
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.module import Module
from torch.nn import parameter
from torch.nn.parameter import Parameter
from torch.nn import init
import math
import copy
from models.preresnet import PreResNet
from models.wide_resnet import WideResNet

# stores the architecture base class argument names
# the name of the linear layer of the architecture
MODEL_ATTRIB_DICT = {'PreResNet' : {'args': ['num_classes', 'depth'], 
                                 'lin_layer': 'fc',
                                 'class': PreResNet},
                    'WideResNet' : {'args': ['num_classes', 'depth', 'widen_factor', 'dropout_rate'], 
                                    'lin_layer': 'linear', 
                                    'class': WideResNet},}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
def BayesFCNet(**kwargs):
    
    base_class = Module
    if 'encoded' in kwargs['net_type']:
        base_class = Module
    else:
        archi_name = ''
        for archi in MODEL_ATTRIB_DICT.keys():
            if archi in kwargs['net_type']:
                if archi_name != '':
                    print('Confusing "net_type" =', kwargs['net_type'])
                archi_name = archi
                base_class = MODEL_ATTRIB_DICT[archi_name]['class']

    
    class bayesFCNet(base_class):

        def __init__(self, **kwargs):

            if 'encoded' in kwargs['net_type']:
                Module.__init__(self)
                # find a better way to let the encoder model know
                # the incoming data feature size
                self.feature_size = kwargs['num_classes']
            else:
                base_class = MODEL_ATTRIB_DICT[archi_name]['class']
                base_init_args = MODEL_ATTRIB_DICT[archi_name]['args']
                linear_layer_name = MODEL_ATTRIB_DICT[archi_name]['lin_layer']

                base_init_kwargs = dict([(arg, kwargs[arg]) for arg in base_init_args if arg in kwargs])
                args_not_provided = [arg for arg in base_init_args if arg not in kwargs]
                if len(args_not_provided) > 0:
                    print("Following arguments for", archi_name, "not provided :", ' '.join(args_not_provided))
                
                base_class.__init__(self, **base_init_kwargs)
                self.feature_size = self._modules[linear_layer_name].weight.size()[1]
                self._modules[linear_layer_name] = Identity()

            self.net_type = kwargs['net_type']
            self.device = kwargs['device']
            self.loss_mixing_ratio = kwargs.get('loss_mixing_ratio', None)

            fc_setup = kwargs['fc_setup']
            num_classes = kwargs['num_classes']
            rendFeature_rank_reduction = kwargs.get('rendFeature_rank_reduction', None)

            if 'trainloader' in kwargs:
                self.dataloader = kwargs['trainloader']

            if fc_setup==[] and self.feature_size != num_classes:
                print("Fully connected setup not provided, defaulting to 1 layer")
                fc_setup = [num_classes]

            if len(fc_setup) > 0:
                if fc_setup[-1] != num_classes:
                    print("Last layer dimension does not match number of class, appending layer at the end!")
                    fc_setup.append(num_classes)

                if 'randFeature' in self.net_type:
                    random_feature_size = int(self.feature_size * (-math.log(accuracy)/accuracy**2))
                    random_feature_size = int(random_feature_size/rendFeature_rank_reduction)
                    print("Random feature size according to accuracy %f is %d" %(accuracy, random_feature_size))
                    self.rand_W = torch.randn((self.feature_size, random_feature_size), device=self.device)
                    self.rand_B = 2*math.pi*torch.rand(random_feature_size, device=self.device)
                    self.feature_size = random_feature_size

                if 'bayesian' not in self.net_type:

                    prev_layer = self.feature_size
                    for i in range(len(fc_setup)):
                        layer = fc_setup[i]
                        w_name = 'fc'+ str(i) + '_w'
                        b_name = 'fc'+ str(i) + '_b'

                        self.register_parameter(b_name, Parameter(torch.Tensor(layer)))
                        self.register_parameter(w_name, Parameter(torch.Tensor(prev_layer, layer)))
                        prev_layer = layer

                    param_dict = dict(self.named_parameters())
                    prev_layer = self.feature_size
                    for i in range(len(fc_setup)):
                        layer = fc_setup[i]
                        stdv = 1. / math.sqrt(prev_layer)
                        w_name = 'fc'+ str(i) + '_w'
                        b_name = 'fc'+ str(i) + '_b'
                        param_dict[w_name].data.uniform_(-stdv, stdv)
                        param_dict[b_name].data.uniform_(-stdv, stdv)
                        prev_layer = layer

                else:
                    prev_layer = self.feature_size
                    for i in range(len(fc_setup)):
                        layer = fc_setup[i]
                        w_mu, b_mu = ['fc'+str(i)+'_w_mu', 'fc'+str(i)+'_b_mu']
                        w_sigma, b_sigma = ['fc'+str(i)+'_w_sigma', 'fc'+str(i)+'_b_sigma']

                        self.register_parameter(w_mu, Parameter(torch.Tensor(prev_layer, layer)))
                        self.register_parameter(w_sigma, Parameter(torch.Tensor(prev_layer, layer)))
                        self.register_parameter(b_mu, Parameter(torch.Tensor(layer)))
                        self.register_parameter(b_sigma, Parameter(torch.Tensor(layer)))
                        prev_layer = layer 

                    param_dict = dict(self.named_parameters())
                    prev_layer = self.feature_size
                    for i in range(len(fc_setup)):
                        layer = fc_setup[i]
                        w_mu, b_mu = ['fc'+str(i)+'_w_mu', 'fc'+str(i)+'_b_mu']
                        w_sigma, b_sigma = ['fc'+str(i)+'_w_sigma', 'fc'+str(i)+'_b_sigma']

                        stdv = 1. / math.sqrt(prev_layer)
                        param_dict[b_mu].data.uniform_(-stdv, stdv)
                        param_dict[b_sigma].data.uniform_(math.log(stdv)-0.5, math.log(stdv))
                        param_dict[w_mu].data.uniform_(-stdv, stdv)
                        param_dict[w_sigma].data.uniform_(math.log(stdv)-0.5, math.log(stdv))
                        prev_layer = layer

                    if 'rank' in self.net_type:
                        index = self.net_type.find('rank_')
                        rank = int(self.net_type[index+5:])
                        self.rank = rank
                        print("Constructing variational variance dependency of rank :",rank)

                        prev_layer = self.feature_size
                        for i in range(len(fc_setup)):
                            layer = fc_setup[i]
                            w_sigma_B, b_sigma_B = ['fc'+str(i)+'_w_sigma_B', 'fc'+str(i)+'_b_sigma_B']
                            self.register_parameter(b_sigma_B, Parameter(torch.Tensor(layer, rank)))
                            self.register_parameter(w_sigma_B, Parameter(torch.Tensor(prev_layer, layer, rank)))
                            prev_layer = layer

                        param_dict = dict(self.named_parameters())
                        prev_layer = self.feature_size
                        for i in range(len(fc_setup)):
                            layer = fc_setup[i]
                            w_sigma_B, b_sigma_B = ['fc'+str(i)+'_w_sigma_B', 'fc'+str(i)+'_b_sigma_B']
                            stdv = 1. / math.sqrt(prev_layer * rank)
                            param_dict[b_sigma_B].data.uniform_(-stdv, stdv)
                            param_dict[w_sigma_B].data.uniform_(-stdv, stdv)
                            prev_layer = layer


            elif "bayesian" in self.net_type:
                raise Exception("FC-Bayesian network initialized with no FC layer! Rerun with FC layer setup")

            self.parameter_names = [name for name in self.named_parameters()]
            self.fc_setup = fc_setup


        def extract_feature(self, x):

            if 'encoded' not in self.net_type:
                return super(bayesFCNet, self).forward(x)
            else:
                return x

        def FC_encode(self, x):

            param_dict = dict(self.named_parameters())

            if 'randFeature' in self.net_type:
                x = math.sqrt(2.0 / self.feature_size) * torch.cos(F.linear(x, self.rand_W.t(), self.rand_B))

            if 'bayesian' not in self.net_type:

                for i in range(len(self.fc_setup)):
                    if i > 0:
                        x = F.relu(x)
                    w_name = 'fc'+ str(i) + '_w'
                    b_name = 'fc'+ str(i) + '_b'
                    w = param_dict[w_name]
                    b = param_dict[b_name]

                    x = F.linear(x, w.t(), b)

            elif 'bayesian' in self.net_type:
                if 'rank' in self.__dict__.keys():
                    noise_ndiag = torch.randn((self.rank), device=self.device)

                prev_layer = self.feature_size
                for i in range(len(self.fc_setup)):

                    if i > 0:
                        x = F.relu(x)

                    layer = self.fc_setup[i]
                    noise_w = torch.randn((prev_layer, layer), device=self.device)
                    noise_b = torch.randn((layer), device=self.device)
                    w_mu = param_dict['fc'+str(i)+'_w_mu']
                    w_sigma = param_dict['fc'+str(i)+'_w_sigma']
                    b_mu = param_dict['fc'+str(i)+'_b_mu']
                    b_sigma = param_dict['fc'+str(i)+'_b_sigma']

                    w = w_mu + torch.exp(w_sigma) * noise_w
                    b = b_mu + torch.exp(b_sigma) * noise_b
                    if 'rank' in self.__dict__.keys():
                        w_sigma_B = param_dict['fc'+str(i)+'_w_sigma_B']
                        b_sigma_B = param_dict['fc'+str(i)+'_b_sigma_B']
                        w = w + torch.matmul(w_sigma_B, noise_ndiag)
                        b = b + torch.matmul(b_sigma_B, noise_ndiag)
                    x = F.linear(x, w.t(), b)
                    prev_layer = layer

            return x


        def avg_encode(self, x, noise_sample, param_list=[]):

            x = self.extract_feature(x)

            if len(param_list)==0:
                final_encode = F.softmax(self.FC_encode(x), dim=1)
                if 'bayesian' in self.net_type:
                    for i in range(noise_sample - 1):
                        final_encode = final_encode + F.softmax(self.FC_encode(x), dim=1)
                    final_encode /= (noise_sample * 1.0)
            else:
                num_param = len(param_list)
                holder = x.repeat(num_param, 1, 1)
                if len(param_list[0].shape)==1:
                    last_index = 0
                    prev_layer = self.feature_size
                    for i in range(len(self.fc_setup)):
                        if i > 0:
                            holder = [F.relu(item) for item in holder]
                        layer = self.fc_setup[i]
                        temp = torch.zeros(num_param, x.size()[0], layer).to(self.device)
                        for j in range(num_param):
                            flattened_w = param_list[j][last_index:last_index+(prev_layer*layer)]
                            w_data = torch.reshape(torch.tensor(flattened_w).to(self.device), (prev_layer, layer))
                            flattened_b = param_list[j][last_index+(prev_layer*layer):last_index+(prev_layer*layer)+(layer)]
                            b_data = torch.reshape(torch.tensor(flattened_b).to(self.device), (layer,))
                            temp[j] = torch.matmul(holder[j], w_data.float()) + b_data.float()
                            print(temp[0], temp[1])
                        holder = temp
                        last_index += prev_layer*layer + layer

                holder = F.softmax(holder, dim=2)
                if (holder[0] == holder[-1]).all():
                    print("All param elements might be same! something is wrong!")
                    print(holder[0], holder[1], holder[-1])
                final_encode = holder.mean(0)

            return final_encode


        def forward(self, x, labs, random_sample_train):

            kl = 0
            input_size = x.size()[0]
            if 'bayesian' not in self.net_type:
                random_sample_train = 1

            x = self.extract_feature(x)

            for i in range(random_sample_train):
                last_layer = self.FC_encode(x)
                probs = F.softmax(last_layer, dim=1)
                class_logprob = torch.log(torch.gather(probs, dim=1, index=labs.reshape((input_size, 1))))

                one_hot_mat = torch.zeros(probs.size()).to(self.device).scatter(1, labs.reshape((input_size, 1)), 1)
                kl -= (1 - self.loss_mixing_ratio) * torch.sum((one_hot_mat - probs)**2).item()/(1.0*probs.size()[1]*probs.size()[0])

                kl -= self.loss_mixing_ratio * torch.sum(class_logprob)/(random_sample_train)

            return kl/(1.0*input_size)


        def return_init_MCMC_state(self):

            param_dict = dict(self.named_parameters())
            if len(self.fc_setup) == 0:
                raise Exception('No FC layer detected!')
            else:
                flattened_state = []
                prev_layer = self.feature_size
                for i in range(len(self.fc_setup)):

                    layer = self.fc_setup[i]
                    stdv = 1. / math.sqrt(prev_layer)
                    w_name = 'fc'+ str(i) + '_w'
                    b_name = 'fc'+ str(i) + '_b'
                    w = param_dict[w_name]
                    b = param_dict[b_name]
                    flattened_w = np.reshape(w.data.detach().cpu().numpy(), (prev_layer*layer))
                    flattened_b = np.reshape(b.data.detach().cpu().numpy(), (layer))
                    flattened_state = np.concatenate((flattened_state, flattened_w, flattened_b))

            return flattened_state


        def pot_energy_FC(self, pos):

            loss = 0
            param_dict = dict(self.named_parameters())
            if len(self.fc_setup) == 0:
                raise Exception('No FC layer detected!')
            else:
                last_index = 0
                prev_layer = self.feature_size
                label_size = 0
                for i in range(len(self.fc_setup)):

                    layer = self.fc_setup[i]
                    flattened_w = pos[last_index:last_index+(prev_layer*layer)]
                    last_index += prev_layer*layer
                    w_data = torch.reshape(torch.tensor(flattened_w), (prev_layer, layer))
                    flattened_b = pos[last_index:last_index+(layer)]
                    last_index += layer
                    b_data = torch.reshape(torch.tensor(flattened_b), (layer,))
                    w_name = 'fc'+ str(i) + '_w'
                    b_name = 'fc'+ str(i) + '_b'
                    param_dict[w_name].data = w_data.type(torch.FloatTensor)
                    param_dict[b_name].data = b_data.type(torch.FloatTensor)
                    param_dict[w_name].data = param_dict[w_name].data.to(utils.device)
                    param_dict[b_name].data = param_dict[b_name].data.to(utils.device)

                for index, data in enumerate(self.dataloader, 0):

                    inputs, labels = data
                    if index==0:
                        label_size = labels.size()
                    inputs = inputs.to(utils.device)
                    labels = labels.to(utils.device)
                    loss += self.forward(inputs, labels, utils.random_sample_train).item()

                import gc
                gc.collect()
                # print("final value of loss is at forward", loss/index)
                return (loss/index + (pos**2).sum()/(2*label_size[0]*index))


        def grad_pot_energy_FC(self, pos):

            final_loss = 0
            flattened_grad = np.zeros(len(pos))
            param_dict = dict(self.named_parameters())
            if len(self.fc_setup) == 0:
                raise Exception('No FC layer detected!')
            else:
                last_index = 0
                prev_layer = self.feature_size
                for i in range(len(self.fc_setup)):

                    layer = self.fc_setup[i]
                    flattened_w = pos[last_index:last_index+(prev_layer*layer)]
                    last_index += prev_layer*layer
                    w_data = torch.reshape(torch.tensor(flattened_w), (prev_layer, layer))
                    flattened_b = pos[last_index:last_index+(layer)]
                    last_index += layer
                    b_data = torch.reshape(torch.tensor(flattened_b), (layer,))
                    w_name = 'fc'+ str(i) + '_w'
                    b_name = 'fc'+ str(i) + '_b'
                    param_dict[w_name].data = w_data.type(torch.FloatTensor)
                    param_dict[b_name].data = b_data.type(torch.FloatTensor)
                    param_dict[w_name].data = param_dict[w_name].data.to(utils.device)
                    param_dict[b_name].data = param_dict[b_name].data.to(utils.device)

                for index, data in enumerate(self.dataloader, 0):

                    inputs, labels = data
                    if index==0:
                        label_size = labels.size()
                    inputs = inputs.to(utils.device)
                    labels = labels.to(utils.device)
                    loss = self.forward(inputs, labels, utils.random_sample_train)
                    loss.backward()
                    final_loss += loss.item()
                    temp_flattened_grad = []
                    prev_layer = self.feature_size
                    for i in range(len(self.fc_setup)):

                        layer = self.fc_setup[i]
                        w_name = 'fc'+ str(i) + '_w'
                        b_name = 'fc'+ str(i) + '_b'
                        w_grad = param_dict[w_name].grad.cpu()
                        b_grad = param_dict[b_name].grad.cpu()
                        w_grad, b_grad = w_grad.detach().numpy(), b_grad.detach().numpy()
                        flattened_w = np.reshape(w_grad, (prev_layer*layer))
                        flattened_b = np.reshape(b_grad, (layer))
                        temp_flattened_grad = np.concatenate((temp_flattened_grad, flattened_w, flattened_b))
                        flattened_grad += temp_flattened_grad

                import gc
                gc.collect()
                final_loss = final_loss/index + (pos**2).sum()/(2*label_size[0]*index)
                flattened_grad = flattened_grad/index + pos/(label_size[0]*index)
                # loss = self.pot_energy_FC(pos)
                print("final value of loss is ", final_loss)#, flattened_grad[:10])

            return flattened_grad, final_loss

    return bayesFCNet(**kwargs)
