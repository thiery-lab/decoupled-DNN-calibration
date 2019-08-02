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
from models.basic_bayes import BayesFCNet
try:
    import gpytorch
    gpytorch_imported = True
except:
    gpytorch_imported = False

# stores the architecture base class argument names
# the name of the linear layer of the architecture
MODEL_ATTRIB_DICT = {'PreResNet' : {'args': ['num_classes', 'depth'], 
                                 'lin_layer': 'fc',
                                 'class': PreResNet},
                    'WideResNet' : {'args': ['num_classes', 'depth', 'widen_factor', 'dropout_rate'], 
                                    'lin_layer': 'linear', 
                                    'class': WideResNet},}


if gpytorch_imported:
    
    class GPNet(gpytorch.Module):

        def __init__(self, device, kernel_net_type, gp_feature_size=256, 
                     num_classes=10, grid_bounds=(-10., 10.), depth=110, grid_size=64):

            super(GPNet, self).__init__()
            self.feature_extractor = BayesFCNet(device=device, net_type=kernel_net_type,
                                                fc_setup=[], num_classes=gp_feature_size, depth=depth)
            self.gp_layer = GaussianProcessLayer(num_dim=gp_feature_size, grid_bounds=grid_bounds, grid_size=grid_size)
            self.grid_bounds = grid_bounds
            self.num_dim = gp_feature_size
            self.device = device
            self.net_type = kernel_net_type if '+GP' in kernel_net_type else kernel_net_type + 'GP'


        def forward(self, x):
            features = self.feature_extractor.extract_feature(x)
            features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
            res = self.gp_layer(features)
            return res


        def modify_grad(self):
            return


    class GaussianProcessLayer(gpytorch.models.AdditiveGridInducingVariationalGP):

        def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
            super(GaussianProcessLayer, self).__init__(grid_size=grid_size, grid_bounds=[grid_bounds],
                                                       num_dim=num_dim, mixing_params=False, sum_output=False)
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                        math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                    )
                )
            )
            self.mean_module = gpytorch.means.ConstantMean()
            self.grid_bounds = grid_bounds


        def forward(self, x):
            mean = self.mean_module(x)
            covar = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean, covar)


    class GaussianProcessLayer1(gpytorch.models.AbstractVariationalGP):
        def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=grid_size, batch_size=num_dim
            )
            variational_strategy = gpytorch.variational.AdditiveGridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds], num_dim=num_dim,
                variational_distribution=variational_distribution, mixing_params=False, sum_output=False
            )
            super().__init__(variational_strategy)

            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(
                    lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                        math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                    )
                )
            )
            self.mean_module = gpytorch.means.ConstantMean()
            self.grid_bounds = grid_bounds

