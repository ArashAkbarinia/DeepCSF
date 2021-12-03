"""
A collection of architectures to do the contrast discrimination task.
"""

import sys
import torch
import torch.nn as nn

from . import pretrained_models


class ContrastDiscrimination(nn.Module):
    def __init__(self, architecture, target_size, transfer_weights=None):
        super(ContrastDiscrimination, self).__init__()

        num_classes = 2

        checkpoint = None
        # assuming architecture is path
        if transfer_weights is None:
            print('Loading model from %s!' % architecture)
            checkpoint = torch.load(architecture, map_location='cpu')
            architecture = checkpoint['arch']
            transfer_weights = checkpoint['transfer_weights']

        model = pretrained_models.get_pretrained_model(architecture, transfer_weights)
        if '_scratch' in architecture:
            architecture = architecture.replace('_scratch', '')
        model = pretrained_models.get_backbone(architecture, model)

        layer = -1
        if len(transfer_weights) >= 2:
            layer = transfer_weights[1]

        if ('deeplabv3_' in architecture or 'fcn_' in architecture
                or 'deeplab' in architecture
                or 'resnet' in architecture or 'resnext' in architecture
        ):
            features, org_classes = pretrained_models._resnet_features(
                model, architecture, layer
            )
        else:
            sys.exit('Unsupported network %s' % architecture)
        self.features = features

        # the numbers for fc layers are hard-coded according to 256 size.
        scale_factor = (target_size / 256)
        self.fc = nn.Linear(int(org_classes * scale_factor), num_classes)

        if checkpoint is not None:
            self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x0, x1):
        x0 = self.features(x0)
        x0 = x0.view(x0.size(0), -1)
        x1 = self.features(x1)
        x1 = x1.view(x1.size(0), -1)
        x = torch.cat([x0, x1], dim=1)
        x = self.fc(x)
        return x
