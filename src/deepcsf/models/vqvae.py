"""
A collection of VAE models.
"""

import abc
import numpy as np
import logging

import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data

from . import nearest_embed
from . import pretrained_features


class AbstractAutoEncoder(nn.Module):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def encode(self, x):
        return

    @abc.abstractmethod
    def decode(self, z):
        return

    @abc.abstractmethod
    def forward(self, x):
        """model return (reconstructed_x, *)"""
        return

    @abc.abstractmethod
    def sample(self, size):
        """sample new images from model"""
        return

    @abc.abstractmethod
    def loss_function(self, **kwargs):
        """accepts (original images, *) where * is the same as returned
        from forward()"""
        return

    @abc.abstractmethod
    def latest_losses(self):
        """returns the latest losses in a dictionary. Useful for logging."""
        return


class VAE(nn.Module):
    """Variational AutoEncoder for MNIST
       Taken from pytorch/examples:
       https://github.com/pytorch/examples/tree/master/vae"""

    def __init__(self, kl_coef=1, **kwargs):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.kl_coef = kl_coef
        self.bce = 0
        self.kl = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, 20)
        if self.cuda():
            sample = sample.cuda()
        sample = self.decode(sample).cpu()
        return sample

    def loss_function(self, x, recon_x, mu, logvar):
        self.bce = F.binary_cross_entropy(recon_x, x.view(-1, 784),
                                          size_average=False)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return self.bce + self.kl_coef * self.kl

    def latest_losses(self):
        return {'bce': self.bce, 'kl': self.kl}


class VQ_VAE(nn.Module):
    """Vector Quantized AutoEncoder for mnist"""

    def __init__(self, hidden=200, k=10, vq_coef=0.2, comit_coef=0.4, **kwargs):
        super(VQ_VAE, self).__init__()

        self.emb_size = k
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, hidden)
        self.fc3 = nn.Linear(hidden, 400)
        self.fc4 = nn.Linear(400, 784)

        self.emb = nearest_embed.NearestEmbed(k, self.emb_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.vq_coef = vq_coef
        self.comit_coef = comit_coef
        self.hidden = hidden
        self.ce_loss = 0
        self.vq_loss = 0
        self.commit_loss = 0

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.fc2(h1)
        return h2.view(-1, self.emb_size, int(self.hidden / self.emb_size))

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.tanh(self.fc4(h3))

    def forward(self, x):
        z_e = self.encode(x.view(-1, 784))
        z_q, _ = self.emb(z_e, weight_sg=True).view(-1, self.hidden)
        emb, _ = self.emb(z_e.detach()).view(-1, self.hidden)
        return self.decode(z_q), z_e, emb

    def sample(self, size):
        sample = torch.randn(size, self.emb_size,
                             int(self.hidden / self.emb_size))
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        sample = self.decode(emb(sample).view(-1, self.hidden)).cpu()
        return sample

    def loss_function(self, x, recon_x, z_e, emb):
        self.ce_loss = F.binary_cross_entropy(recon_x, x.view(-1, 784))
        self.vq_loss = F.mse_loss(emb, z_e.detach())
        self.commit_loss = F.mse_loss(z_e, emb.detach())

        return self.ce_loss + self.vq_coef * self.vq_loss + self.comit_coef * self.commit_loss

    def latest_losses(self):
        return {'cross_entropy': self.ce_loss, 'vq': self.vq_loss,
                'commitment': self.commit_loss}


class ResBlock(nn.Module):
    def __init__(self, in_chns, out_chns, mid_chns=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_chns is None:
            mid_chns = out_chns

        self.in_chns = in_chns
        self.out_chns = out_chns
        layers = [
            nn.ReLU(),
            nn.Conv2d(in_chns, mid_chns, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_chns, out_chns, kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_chns))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        if self.in_chns == self.out_chns:
            return x + self.convs(x)
        else:
            return self.convs(x)


class CVAE(AbstractAutoEncoder):
    def __init__(self, d, kl_coef=0.1, num_chns=3, **kwargs):
        super(CVAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(num_chns, d // 2, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d // 2, d, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
        )
        self.decoder = nn.Sequential(
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),

            nn.ConvTranspose2d(d, d // 2, kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(d // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d // 2, 3, kernel_size=4, stride=2, padding=1,
                               bias=False),
        )
        self.f = 8
        self.d = d
        self.fc11 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.fc12 = nn.Linear(d * self.f ** 2, d * self.f ** 2)
        self.kl_coef = kl_coef
        self.kl_loss = 0
        self.mse = 0

    def encode(self, x):
        h1 = self.encoder(x)
        h1 = h1.view(-1, self.d * self.f ** 2)
        return self.fc11(h1), self.fc12(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        z = z.view(-1, self.d, self.f, self.f)
        h3 = self.decoder(z)
        return torch.tanh(h3)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def sample(self, size):
        sample = torch.randn(size, self.d * self.f ** 2, requires_grad=False)
        if self.cuda():
            sample = sample.cuda()
        return self.decode(sample).cpu()

    def loss_function(self, x, recon_x, mu, logvar):
        self.mse = F.mse_loss(recon_x, x)
        batch_size = x.size(0)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        self.kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        self.kl_loss /= batch_size * 3 * 1024

        # return mse
        return self.mse + self.kl_coef * self.kl_loss

    def latest_losses(self):
        return {'mse': self.mse, 'kl': self.kl_loss}


class HueLoss(torch.nn.Module):
    def forward(self, recon_x, x):
        ret = recon_x - x
        ret[ret > 1] -= 2
        ret[ret < -1] += 2
        ret = ret ** 2
        return torch.mean(ret)


class SegLoss(torch.nn.Module):
    def forward(self, recon_x, x):
        ret = recon_x - x
        ret[ret != 0] = 1
        ret = ret ** 2
        return torch.mean(ret)


class ResNet_VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, resnet=None, vq_coef=1, commit_coef=0.5,
                 in_chns=3, colour_space='rgb', out_chns=None, task=None,
                 **kwargs):
        super(ResNet_VQ_CVAE, self).__init__()

        if out_chns is None:
            out_chns = in_chns

        self.colour_space = colour_space
        self.task = task
        self.hue_loss = HueLoss()

        self.resnet = resnet
        self.encoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, d, kernel_size=3, padding=1),
            nn.BatchNorm2d(d),
        )
        from torchvision.models.segmentation.deeplabv3 import ASPP
        self.decoder = nn.Sequential(
            ASPP(d, d, [12, 24, 36]),
            nn.Conv2d(d, d, 3, padding=1, bias=False),
        )
        if self.task == 'segmentation':
            self.fc = nn.Sequential(
                nn.BatchNorm2d(d),
                nn.ReLU(),
                nn.Conv2d(d, out_chns, 1)
            )
        self.d = d
        self.emb = nearest_embed.NearestEmbed(k, d)
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if l in resnet.modules():
                continue
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        features = self.resnet(x)
        x = features["out"]
        return self.encoder(x)

    def decode(self, x, input_shape):
        if self.task == 'segmentation':
            x = self.fc(self.decoder(x))
            x = F.interpolate(
                x, size=input_shape, mode='bilinear', align_corners=False
            )
            return x
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        input_shape = x.shape[-2:]
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q, input_shape), z_e, emb, argmin

    def sample(self, size):
        sample = torch.randn(size, self.d, self.f, self.f, requires_grad=False),
        if self.cuda():
            sample = sample.cuda()
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.d, self.f, self.f)).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        if self.colour_space == 'hsv':
            self.mse = F.mse_loss(recon_x[:, 1:], x[:, 1:])
            self.mse += self.hue_loss(recon_x[:, 0], x[:, 0])
        elif self.task == 'segmentation':
            self.mse = F.cross_entropy(recon_x, x, ignore_index=255)
        else:
            self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1))

        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        return {'mse': self.mse, 'vq': self.vq_loss,
                'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)


class VQ_CVAE(nn.Module):
    def __init__(self, d, k=10, kl=None, bn=True, vq_coef=1, commit_coef=0.5,
                 in_chns=3, colour_space='rgb', out_chns=None, task=None,
                 cos_distance=False, use_decor_loss=0, **kwargs):
        super(VQ_CVAE, self).__init__()

        if out_chns is None:
            out_chns = in_chns
        self.out_chns = out_chns
        if task == 'segmentation':
            out_chns = d
        self.use_decor_loss = use_decor_loss
        if self.use_decor_loss != 0:
            self.decor_loss = torch.zeros(1)

        self.d = d
        self.k = k
        if kl is None:
            kl = d
        self.kl = kl
        self.emb = nearest_embed.NearestEmbed(k, kl, cos_distance)

        self.colour_space = colour_space
        self.task = task
        self.hue_loss = HueLoss()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chns, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            ResBlock(d, d, bn=True),
            nn.BatchNorm2d(d),
            ResBlock(d, kl, bn=True),
            nn.BatchNorm2d(kl),
        )
        self.decoder = nn.Sequential(
            ResBlock(kl, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d, out_chns, kernel_size=4, stride=2, padding=1)
        )
        if self.task == 'segmentation':
            self.fc = nn.Sequential(
                nn.BatchNorm2d(d),
                nn.ReLU(),
                nn.Conv2d(d, self.out_chns, 1)
            )
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        if self.task == 'segmentation':
            return self.fc(self.decoder(x))
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        if self.cuda():
            sample = torch.tensor(
                torch.randn(size, self.kl, self.f, self.f), requires_grad=False
            ).cuda()
        else:
            sample = torch.tensor(
                torch.randn(size, self.kl, self.f, self.f), requires_grad=False
            )
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.kl, self.f, self.f)).cpu()

    def sample_inds(self, inds):
        assert len(inds.shape) == 2
        rows = inds.shape[0]
        cols = inds.shape[1]
        inds = inds.reshape(rows * cols)
        weights = self.emb.weight.detach().cpu().numpy()
        sample = np.zeros((self.kl, rows, cols))
        sample = sample.reshape(self.kl, rows * cols)
        for i in range(self.k):
            which_inds = inds == i
            sample[:, which_inds] = np.broadcast_to(
                weights[:, i], (which_inds.sum(), self.kl)
            ).T
        sample = sample.reshape(self.kl, rows, cols)
        emb = torch.tensor(sample, dtype=torch.float32).unsqueeze(dim=0)
        return self.decode(emb).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        if self.colour_space == 'hsv':
            self.mse = F.mse_loss(recon_x[:, 1:], x[:, 1:])
            self.mse += self.hue_loss(recon_x[:, 0], x[:, 0])
        elif self.task == 'segmentation':
            if self.out_chns == 1:
                self.mse = F.binary_cross_entropy_with_logits(
                    recon_x.squeeze(), x
                )
            else:
                self.mse = F.cross_entropy(recon_x, x, ignore_index=255)
        elif self.colour_space == 'labhue':
            self.mse = F.mse_loss(recon_x[:, :3], x[:, :3])
            self.mse += self.hue_loss(recon_x[:, 3], x[:, 3])
        else:
            self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1))

        if self.use_decor_loss != 0:
            emb_weights = self.emb.weight.detach()
            mean_mat = emb_weights.mean(dim=0)
            emb_weights = emb_weights.sub(mean_mat)
            corr = torch.zeros((self.k, self.k))
            for i in range(self.k - 1):
                for j in range(i + 1, self.k):
                    r_num = emb_weights[:, i].dot(emb_weights[:, j])
                    r_den = torch.norm(emb_weights[:, i], 2) * torch.norm(
                        emb_weights[:, j], 2
                    )
                    current_corr = r_num / r_den
                    corr[i, j] = current_corr
                    corr[j, i] = current_corr
            self.decor_loss = abs(corr).mean()
            if self.use_decor_loss < 0:
                self.decor_loss = 1 - self.decor_loss
            return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss + self.decor_loss

        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        if self.use_decor_loss:
            return {'mse': self.mse, 'vq': self.vq_loss,
                    'commitment': self.commit_loss,
                    'decorr': self.decor_loss}
        return {'mse': self.mse, 'vq': self.vq_loss,
                'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)


class Backbone_VQ_VAE(nn.Module):
    def __init__(self, d, k=10, kl=None, bn=True, vq_coef=1, commit_coef=0.5,
                 in_chns=3, colour_space='rgb', out_chns=None, task=None,
                 cos_distance=False, use_decor_loss=0, backbone=None, **kwargs):
        super(Backbone_VQ_VAE, self).__init__()

        self.backbone_encoder = pretrained_features.ResNetIntermediate(
            **backbone)

        if out_chns is None:
            out_chns = in_chns
        self.out_chns = out_chns
        if task == 'segmentation':
            out_chns = d
        self.use_decor_loss = use_decor_loss
        if self.use_decor_loss != 0:
            self.decor_loss = torch.zeros(1)

        self.d = d
        self.k = k
        if kl is None:
            kl = d
        self.kl = kl
        self.emb = nearest_embed.NearestEmbed(k, kl, cos_distance)

        self.colour_space = colour_space
        self.task = task
        self.hue_loss = HueLoss()

        self.encoder = nn.Sequential(
            self.backbone_encoder,
            ResBlock(self.backbone_encoder.get_num_kernels(), kl, bn=True),
            nn.BatchNorm2d(kl),
        )
        conv_transposes = []
        num_conv_transpose = int(self.backbone_encoder.spatial_ratio / 2)
        for i in range(int(np.log2(num_conv_transpose))):
            conv_transposes.append(
                nn.ConvTranspose2d(d, d, kernel_size=4, stride=2, padding=1)
            )
            conv_transposes.append(nn.BatchNorm2d(d))
            conv_transposes.append(nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(
            ResBlock(kl, d),
            nn.BatchNorm2d(d),
            ResBlock(d, d),
            *conv_transposes,
            nn.ConvTranspose2d(d, out_chns, kernel_size=4, stride=2, padding=1)
        )
        if self.task == 'segmentation':
            self.fc = nn.Sequential(
                nn.BatchNorm2d(d),
                nn.ReLU(),
                nn.Conv2d(d, self.out_chns, 1)
            )
        self.vq_coef = vq_coef
        self.commit_coef = commit_coef
        self.mse = 0
        self.vq_loss = torch.zeros(1)
        self.commit_loss = 0

        for l in self.modules():
            if (
                    isinstance(l, pretrained_features.ResNetIntermediate) or
                    l in self.backbone_encoder.modules()
            ):
                continue
            if isinstance(l, nn.Linear) or isinstance(l, nn.Conv2d):
                l.weight.detach().normal_(0, 0.02)
                torch.fmod(l.weight, 0.04)
                nn.init.constant_(l.bias, 0)

        self.encoder[-1].weight.detach().fill_(1 / 40)

        self.emb.weight.detach().normal_(0, 0.02)
        torch.fmod(self.emb.weight, 0.04)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        if self.task == 'segmentation':
            return self.fc(self.decoder(x))
        return torch.tanh(self.decoder(x))

    def forward(self, x):
        z_e = self.encode(x)
        self.f = z_e.shape[-1]
        z_q, argmin = self.emb(z_e, weight_sg=True)
        emb, _ = self.emb(z_e.detach())
        return self.decode(z_q), z_e, emb, argmin

    def sample(self, size):
        if self.cuda():
            sample = torch.tensor(
                torch.randn(size, self.kl, self.f, self.f), requires_grad=False
            ).cuda()
        else:
            sample = torch.tensor(
                torch.randn(size, self.kl, self.f, self.f), requires_grad=False
            )
        emb, _ = self.emb(sample)
        return self.decode(emb.view(size, self.kl, self.f, self.f)).cpu()

    def sample_inds(self, inds):
        assert len(inds.shape) == 2
        rows = inds.shape[0]
        cols = inds.shape[1]
        inds = inds.reshape(rows * cols)
        weights = self.emb.weight.detach().cpu().numpy()
        sample = np.zeros((self.kl, rows, cols))
        sample = sample.reshape(self.kl, rows * cols)
        for i in range(self.k):
            which_inds = inds == i
            sample[:, which_inds] = np.broadcast_to(
                weights[:, i], (which_inds.sum(), self.kl)
            ).T
        sample = sample.reshape(self.kl, rows, cols)
        emb = torch.tensor(sample, dtype=torch.float32).unsqueeze(dim=0)
        return self.decode(emb).cpu()

    def loss_function(self, x, recon_x, z_e, emb, argmin):
        if self.colour_space == 'hsv':
            self.mse = F.mse_loss(recon_x[:, 1:], x[:, 1:])
            self.mse += self.hue_loss(recon_x[:, 0], x[:, 0])
        elif self.task == 'segmentation':
            if self.out_chns == 1:
                self.mse = F.binary_cross_entropy_with_logits(
                    recon_x.squeeze(), x
                )
            else:
                self.mse = F.cross_entropy(recon_x, x, ignore_index=255)
        elif self.colour_space == 'labhue':
            self.mse = F.mse_loss(recon_x[:, :3], x[:, :3])
            self.mse += self.hue_loss(recon_x[:, 3], x[:, 3])
        else:
            self.mse = F.mse_loss(recon_x, x)

        self.vq_loss = torch.mean(torch.norm((emb - z_e.detach()) ** 2, 2, 1))
        self.commit_loss = torch.mean(
            torch.norm((emb.detach() - z_e) ** 2, 2, 1))

        if self.use_decor_loss != 0:
            emb_weights = self.emb.weight.detach()
            mean_mat = emb_weights.mean(dim=0)
            emb_weights = emb_weights.sub(mean_mat)
            corr = torch.zeros((self.k, self.k))
            for i in range(self.k - 1):
                for j in range(i + 1, self.k):
                    r_num = emb_weights[:, i].dot(emb_weights[:, j])
                    r_den = torch.norm(emb_weights[:, i], 2) * torch.norm(
                        emb_weights[:, j], 2
                    )
                    current_corr = r_num / r_den
                    corr[i, j] = current_corr
                    corr[j, i] = current_corr
            self.decor_loss = abs(corr).mean()
            if self.use_decor_loss < 0:
                self.decor_loss = 1 - self.decor_loss
            return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss + self.decor_loss

        return self.mse + self.vq_coef * self.vq_loss + self.commit_coef * self.commit_loss

    def latest_losses(self):
        if self.use_decor_loss:
            return {'mse': self.mse, 'vq': self.vq_loss,
                    'commitment': self.commit_loss,
                    'decorr': self.decor_loss}
        return {'mse': self.mse, 'vq': self.vq_loss,
                'commitment': self.commit_loss}

    def print_atom_hist(self, argmin):

        argmin = argmin.detach().cpu().numpy()
        unique, counts = np.unique(argmin, return_counts=True)
        logging.info(counts)
        logging.info(unique)
