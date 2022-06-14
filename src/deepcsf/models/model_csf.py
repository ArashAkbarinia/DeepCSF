"""
A collection of architectures to do the contrast discrimination task.
"""

import sys
import torch
import torch.nn as nn

from . import pretrained_models


def _load_csf_model(weights, target_size, net_type, classifier):
    print('Loading CSF test model from %s!' % weights)
    checkpoint = torch.load(weights, map_location='cpu')
    architecture = checkpoint['arch']
    transfer_weights = checkpoint['transfer_weights']

    net_class = ContrastDiscrimination if net_type == 'ContrastDiscrimination' else GratingDetector
    model = net_class(architecture, target_size, transfer_weights, classifier)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    return model


def load_contrast_discrimination(weights, target_size, classifier):
    return _load_csf_model(weights, target_size, 'ContrastDiscrimination', classifier)


def load_grating_detector(weights, target_size, classifier):
    return _load_csf_model(weights, target_size, 'GratingDetector', classifier)


class CSFNetwork(nn.Module):
    def __init__(self, architecture, target_size, transfer_weights, input_nodes, classifier):
        super(CSFNetwork, self).__init__()

        num_classes = 2
        self.input_nodes = input_nodes
        self.architecture = architecture

        model = pretrained_models.get_pretrained_model(architecture, transfer_weights)
        if '_scratch' in architecture:
            architecture = architecture.replace('_scratch', '')
        model = pretrained_models.get_backbone(architecture, model)
        self.in_type = self.set_img_type(model)

        layer = transfer_weights[1] if len(transfer_weights) >= 2 else -1

        if layer == 'fc':
            features = model
            if hasattr(model, 'num_classes'):
                org_classes = model.num_classes
            else:
                last_layer = list(model.children())[-1]
                if type(last_layer) is torch.nn.modules.container.Sequential:
                    org_classes = last_layer[-1].out_features
                else:
                    org_classes = last_layer.out_features
        elif (
                'fcn_' in architecture or 'deeplab' in architecture
                or 'resnet' in architecture or 'resnext' in architecture
                or 'taskonomy_' in architecture
        ):
            features, org_classes = pretrained_models.resnet_features(
                model, architecture, layer, target_size
            )
        elif 'clip' in architecture:
            features, org_classes = pretrained_models.clip_features(model, architecture, layer)
        else:
            sys.exit('Unsupported network %s' % architecture)
        self.features = features

        if classifier == 'nn':
            scale_factor = num_classes * (self.input_nodes / 2)
            self.fc = nn.Linear(int(org_classes * scale_factor), num_classes)
        else:
            self.fc = None  # e.g. for SVM

    def set_img_type(self, model):
        return model.conv1.weight.dtype if 'clip' in self.architecture else torch.float32

    def check_img_type(self, x):
        return x.type(self.in_type) if 'clip' in self.architecture else x

    def extract_features(self, x):
        x = self.features(self.check_img_type(x))
        return x

    def do_classifier(self, x):
        return x if self.fc is None else self.fc(x)


class ContrastDiscrimination(CSFNetwork):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(ContrastDiscrimination, self).__init__(
            architecture, target_size, transfer_weights, 2, classifier
        )

    def forward(self, x0, x1):
        x0 = self.extract_features(x0)
        x0 = x0.view(x0.size(0), -1).float()
        x1 = self.extract_features(x1)
        x1 = x1.view(x1.size(0), -1).float()
        x = torch.cat([x0, x1], dim=1)
        return self.do_classifier(x)


class GratingDetector(CSFNetwork):
    def __init__(self, architecture, target_size, transfer_weights, classifier):
        super(GratingDetector, self).__init__(
            architecture, target_size, transfer_weights, 1, classifier
        )

    def forward(self, x):
        x = self.extract_features(x)
        x = x.view(x.size(0), -1).float()
        return self.do_classifier(x)
