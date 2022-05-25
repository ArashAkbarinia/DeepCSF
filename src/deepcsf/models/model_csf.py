"""
A collection of architectures to do the contrast discrimination task.
"""

import sys
import torch
import torch.nn as nn

from . import pretrained_models


def _load_csf_model(weights, target_size, net_type):
    print('Loading CSF test model from %s!' % weights)
    checkpoint = torch.load(weights, map_location='cpu')
    architecture = checkpoint['arch']
    transfer_weights = checkpoint['transfer_weights']

    net_class = ContrastDiscrimination if net_type == 'ContrastDiscrimination' else GratingDetector
    model = net_class(architecture, target_size, transfer_weights)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_contrast_discrimination(weights, target_size):
    return _load_csf_model(weights, target_size, 'ContrastDiscrimination')


def load_grating_detector(weights, target_size):
    return _load_csf_model(weights, target_size, 'GratingDetector')


class CSFNetwork(nn.Module):
    def __init__(self, architecture, target_size, transfer_weights, scale_factor):
        super(CSFNetwork, self).__init__()

        num_classes = 2
        self.architecture = architecture

        model = pretrained_models.get_pretrained_model(architecture, transfer_weights)
        if '_scratch' in architecture:
            architecture = architecture.replace('_scratch', '')
        model = pretrained_models.get_backbone(architecture, model)

        layer = transfer_weights[1] if len(transfer_weights) >= 2 else -1

        if layer == 'fc':
            features = model
            org_classes = 1000
            scale_factor = 1
        elif (
                'fcn_' in architecture or 'deeplab' in architecture
                or 'resnet' in architecture or 'resnext' in architecture
                or 'taskonomy_' in architecture
        ):
            features, org_classes, scale_factor = pretrained_models.resnet_features(
                model, architecture, layer, target_size
            )
        elif 'clip' in architecture:
            features = model
            if 'B32' in architecture:
                org_classes = 512
            elif 'L14' in architecture:
                org_classes = 768
            scale_factor = 1
        else:
            sys.exit('Unsupported network %s' % architecture)
        self.features = features

        # the numbers for fc layers are hard-coded according to 256 size.
        scale_factor = num_classes * scale_factor
        self.fc = nn.Linear(int(org_classes * scale_factor), num_classes)

    def check_img_type(self, x):
        return x.type(self.features.conv1.weight.dtype) if 'clip' in self.architecture else x

    def extract_features(self, x):
        x = self.features(self.check_img_type(x))
        return x[0] if 'inception' in self.architecture else x


class ContrastDiscrimination(CSFNetwork):
    def __init__(self, architecture, target_size, transfer_weights):
        super(ContrastDiscrimination, self).__init__(architecture, target_size, transfer_weights, 1)

    def forward(self, x0, x1):
        x0 = self.extract_features(x0)
        x0 = x0.view(x0.size(0), -1).float()
        x0 = self.extract_features(x1)
        x1 = x1.view(x1.size(0), -1).float()
        x = torch.cat([x0, x1], dim=1)
        x = self.fc(x)
        return x


class GratingDetector(CSFNetwork):
    def __init__(self, architecture, target_size, transfer_weights):
        super(GratingDetector, self).__init__(architecture, target_size, transfer_weights, 0.5)

    def forward(self, x):
        x = self.extract_features(x)
        x = x.view(x.size(0), -1).float()
        x = self.fc(x)
        return x
