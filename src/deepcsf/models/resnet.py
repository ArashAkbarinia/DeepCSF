"""
Custom version of ResNet
"""

import torch
import torch.nn as nn

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
    'resnet_basic_custom', 'resnet_basic_1111', 'resnet_basic_2111',
    'resnet_basic_2211', 'resnet_basic_2221',
    'resnet_bottleneck_custom', 'resnet_bottleneck_1111',
    'resnet_bottleneck_2111', 'resnet_bottleneck_2211',
    'resnet_bottleneck_2221', 'resnet_bottleneck_2222'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     bias=False)


def conv_avg(in_planes, kernel_size, stride=1, groups=None, dilation=1):
    if groups is None:
        groups = in_planes
    num_pixels = kernel_size * kernel_size * (in_planes / groups)
    initial_value = 1.0 / num_pixels
    conv_kernel = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride,
                            padding=dilation, groups=groups, bias=False,
                            dilation=dilation)
    nn.init.constant_(conv_kernel.weight, initial_value)
    conv_kernel.requires_grad = False
    return conv_kernel


class LocalContrastBlock(nn.Module):

    def __init__(self, planes, kernel_size=3, stride=1, groups=None):
        super(LocalContrastBlock, self).__init__()
        self.conv_average = conv_avg(planes, kernel_size=kernel_size,
                                     stride=stride, groups=groups)

    def forward(self, x):
        x_avg = self.conv_average(x)
        x = x - x_avg
        x = x ** 2
        x = self.conv_average(x)
        # TODO: this can be uncommented and this part should not have no
        #  gradient and in eval mode
        # x = x ** 0.5

        return x


class ContrastPoolingBlock(nn.Module):

    def __init__(self, planes, pooling_type, kernel_size=3, stride=2,
                 padding=1):
        super(ContrastPoolingBlock, self).__init__()
        self.pooling_type = pooling_type
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride,
                                     padding=padding)
        # TODO: merge all reductions to one type
        if self.pooling_type in {'mix', 'contrast'}:
            self.reduction = conv1x1(planes * 2, planes)
        if self.pooling_type in {'contrast_avg', 'contrast_max'}:
            self.reduction3 = conv1x1(planes * 3, planes)
        if 'contrast' in self.pooling_type:
            self.local_contrast = self._local_contrast(planes,
                                                       kernel_size=3, stride=1)
        if self.pooling_type not in {'max', 'avg'}:
            self.bn = nn.BatchNorm2d(planes)

    def _local_contrast(self, planes, kernel_size=3, stride=1):
        layers = []
        layers.append(
            LocalContrastBlock(planes, kernel_size=kernel_size, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.pooling_type == 'none':
            out = x
        elif self.pooling_type == 'max':
            out = self.max_pool(x)
        elif self.pooling_type == 'avg':
            out = self.avg_pool(x)
        else:
            x_max = self.max_pool(x)
            x_avg = self.avg_pool(x)
            if self.pooling_type == 'mix':
                x = torch.cat((x_max, x_avg), dim=1)
                x = self.reduction(x)
            elif self.pooling_type == 'contrast_avg':
                x = self.local_contrast(x)
                x = self.avg_pool(x)
                x = torch.cat((x_max, x_avg, x), dim=1)
                x = self.reduction3(x)
            elif self.pooling_type == 'contrast_max':
                x = self.local_contrast(x)
                x = self.max_pool(x)
                x = torch.cat((x_max, x_avg, x), dim=1)
                x = self.reduction3(x)
            elif self.pooling_type == 'contrast':
                x = self.local_contrast(x)
                x = self.avg_pool(x)
                x = x_max * x + x_avg * (1 - x)
            out = self.bn(x)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000,
                 zero_init_residual=False,
                 groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None, pooling_type='max', in_chns=3,
                 inplanes=64, kernel_size=7, stride=2):
        # TODO: pass pooling_type as an argument
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.in_chns = in_chns
        self.inplanes = inplanes
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation
                )
            )
        self.groups = groups
        self.base_width = width_per_group
        # FIXME: bad hack for CIFAR and MNIST, please fix me
        if num_classes == 10:
            self.conv1 = nn.Conv2d(
                self.in_chns, self.inplanes, kernel_size=3, stride=1, padding=1,
                bias=False
            )
            pooling_type = 'none'
        else:
            self.conv1 = nn.Conv2d(
                self.in_chns, self.inplanes, kernel_size=kernel_size, stride=2,
                padding=int(kernel_size / 2), bias=False
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.contrast_pool = self._contrast_pooling(
            self.inplanes, pooling_type, kernel_size=3, stride=2, padding=1
        )
        self.layer1 = self._make_layer(block, self.inplanes, layers[0])
        self.layer2 = self._make_layer(
            block, inplanes * 2, layers[1], stride=stride,
            dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, inplanes * 4, layers[2], stride=stride,
            dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, inplanes * 8, layers[3], stride=stride,
            dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        fcm = 8
        if layers[0] == 0:
            raise ValueError('At least one layer should have planes')
        elif layers[1] == 0:
            fcm = 1
        elif layers[2] == 0:
            fcm = 2
        elif layers[3] == 0:
            fcm = 4
        self.fc = nn.Linear(inplanes * fcm * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        layers = []
        # if None, nothing in this layer
        if blocks > 0:
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers.append(
                block(
                    self.inplanes, planes, stride, downsample, self.groups,
                    self.base_width, previous_dilation, norm_layer
                )
            )
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(
                    block(
                        self.inplanes, planes, groups=self.groups,
                        base_width=self.base_width, dilation=self.dilation,
                        norm_layer=norm_layer
                    )
                )

        return nn.Sequential(*layers)

    def _contrast_pooling(self, planes, pooling_type, kernel_size=3, stride=2,
                          padding=1):
        layers = []
        layers.append(
            ContrastPoolingBlock(
                planes, pooling_type, kernel_size=kernel_size,
                stride=stride, padding=padding
            )
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.contrast_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(arch, block_type, planes, pretrained, progress, **kwargs):
    model = ResNet(block_type, planes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet_basic_custom(planes, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Basic-Custom model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # TODO: should we pass arch name in case of URL?
    return _resnet('', BasicBlock, planes, pretrained, progress, **kwargs)


def resnet_basic_1111(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Basic-1111 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_basic_1111', BasicBlock, [1, 1, 1, 1], pretrained,
                   progress, **kwargs)


def resnet_basic_2111(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Basic-2111 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_basic_2111', BasicBlock, [2, 1, 1, 1], pretrained,
                   progress, **kwargs)


def resnet_basic_2211(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Basic-2211 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_basic_2211', BasicBlock, [2, 2, 1, 1], pretrained,
                   progress, **kwargs)


def resnet_basic_2221(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Basic-2221 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_basic_2221', BasicBlock, [2, 2, 2, 1], pretrained,
                   progress, **kwargs)


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet_bottleneck_custom(planes, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Basic-Custom model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # TODO: should we pass arch name in case of URL?
    return _resnet('', Bottleneck, planes, pretrained, progress, **kwargs)


def resnet_bottleneck_1111(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Bottleneck-1111 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_bottleneck_1111', Bottleneck, [1, 1, 1, 1],
                   pretrained, progress, **kwargs)


def resnet_bottleneck_2111(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Bottleneck-2111 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_bottleneck_2111', Bottleneck, [2, 1, 1, 1],
                   pretrained, progress, **kwargs)


def resnet_bottleneck_2211(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Bottleneck-2211 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_bottleneck_2211', Bottleneck, [2, 2, 1, 1],
                   pretrained, progress, **kwargs)


def resnet_bottleneck_2221(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Bottleneck-2221 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_bottleneck_2221', Bottleneck, [2, 2, 2, 1],
                   pretrained, progress, **kwargs)


def resnet_bottleneck_2222(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-Bottleneck-2222 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet_bottleneck_2222', Bottleneck, [2, 2, 2, 2],
                   pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained=False, progress=True, **kwargs)


def resnext101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained=False, progress=True, **kwargs)
