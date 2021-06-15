"""
Extracting features from different layers of a pretrained model.
"""

import sys

import torch.nn as nn

from torchvision import models as classification
from torchvision.models import segmentation
from torchvision.models import resnet as presnet

from . import model_utils


class ResNetIntermediate(nn.Module):
    def __init__(self, arch_name, layer_name, layer_type='relu',
                 weights_path=None):
        """
        Extract features from a specific layer of ResNet architecture.
        :param arch_name: the name of the architecture
        :param layer_name: in this format <area>.<block>.<layer> e.g. 1.0.1
               layers are indexed from 1, area and block from 0
        :param layer_type: extracting features after conv, batch normalisation,
               relu or max.
        """
        super(ResNetIntermediate, self).__init__()

        if layer_type not in ['conv', 'bn', 'relu', 'max']:
            sys.exit(
                'ResNetLayerActivation: conv_bn_relu %s is not supported' %
                layer_type
            )

        # loading the pretrained network
        model = get_pretrained_model(arch_name)
        # extracting the feature part of the network common across all resnets
        model = get_backbones(arch_name, model)

        self.relu = None
        self.last_block = None
        self.last_layer = None

        entire_area = ['area%d' % e for e in range(0, 5)]
        if 'deeplab' in arch_name or 'fcn_' in arch_name:
            spatial_ratios = [4, 4, 8, 8, 8]
        else:
            spatial_ratios = [4, 4, 8, 16, 32]
        if layer_name in entire_area:
            # easy case of no sub blocks and layers must be identified
            print('Activation for the entire %s' % layer_name)
            area_inds = [4, 5, 6, 7, 8]
            area_num = int(layer_name[-1])
            last_area_ind = area_inds[area_num]
            self.features = nn.Sequential(
                *list(model.children())[:last_area_ind]
            )
            self.spatial_ratio = spatial_ratios[area_num]
        else:
            layer_parts = layer_name.split('.')
            last_block = None
            last_layer = None
            area_num = int(layer_parts[0])
            block_num = int(layer_parts[1])
            layer_num = int(layer_parts[2])
            if layer_num < 1:
                sys.exit('Layer num must be bigger than 0 %s' % layer_name)
            area_inds = [1, 4, 5, 6, 7]
            last_area_ind = area_inds[area_num]
            # area 0 is special since it doesn't have blocks
            if area_num > 0:
                if layer_type == 'relu':
                    self.relu = nn.ReLU(inplace=True)
                    which_fun = model_utils.resnet_bn_layer
                elif layer_type == 'bn':
                    which_fun = model_utils.resnet_bn_layer
                else:
                    which_fun = model_utils.resnet_conv_layer
                last_area = list(model.children())[last_area_ind]
                last_block, last_layer = which_fun(
                    last_area, block_num, layer_num
                )
                self.features = nn.Sequential(
                    *list(model.children())[:last_area_ind]
                )
                # after the first block the spatial resolution is already halved
                if block_num > 0:
                    self.spatial_ratio = spatial_ratios[area_num + 1]
                else:
                    self.spatial_ratio = spatial_ratios[area_num]
            else:
                self.spatial_ratio = 2
                if layer_type == 'bn':
                    last_area_ind = 2
                elif layer_type == 'relu':
                    last_area_ind = 3
                elif layer_type == 'max':
                    last_area_ind = 4
                    self.spatial_ratio = 4
                self.features = nn.Sequential(
                    *list(model.children())[:last_area_ind]
                )
            self.last_block = last_block
            self.last_layer = last_layer

    def get_num_kernels(self):
        if self.last_layer is None:
            features = list(self.features.children())[::-1]
            if isinstance(features[0], nn.Sequential):
                last_block = list(features[0].children())[-1]
                if isinstance(last_block, presnet.Bottleneck):
                    return last_block.conv3.out_channels
                else:
                    return last_block.conv2.out_channels
            for f in features:
                if type(f) is nn.Conv2d:
                    return f.out_channels
        else:
            for f in list(self.last_layer.children())[::-1]:
                if type(f) is nn.Conv2d:
                    return f.out_channels

    def forward(self, x):
        x = self.features(x)
        if self.last_block is not None:
            x = self.last_block(x)
        if self.last_layer is not None:
            x = self.last_layer(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def get_pretrained_model(network_name):
    if 'deeplab' in network_name or 'fcn_' in network_name:
        model = segmentation.__dict__[network_name](pretrained=False)
    else:
        model = classification.__dict__[network_name](pretrained=False)
    return model


def get_backbones(network_name, model):
    return model
