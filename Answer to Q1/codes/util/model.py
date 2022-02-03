import torch
from torch.nn import Module, Linear, ReLU, Dropout, Sequential, Conv2d
from torchvision.models import resnet18


class FeatureExtractor(Module):
    # Stage 1: Feature Extractor
    # Adopted ResNet18
    def __init__(self, num_features):
        super(FeatureExtractor, self).__init__()
        self.extractor = resnet18(pretrained=False)
        default_feat = self.extractor.fc.in_features
        self.extractor.conv1 = Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.extractor.fc = Linear(default_feat, num_features)
        self.relu = ReLU()

    def forward(self, x):
        x = self.extractor(x)
        x = self.relu(x)
        return x


class PoolingFilter(Module):
    # Stage 2: MIL Pooling Filter
    def __init__(self, num_bins,
                 sigma=None, alpha=None, beta=None):
        super(PoolingFilter, self).__init__()
        sigma = sigma or 0.05
        alpha = alpha or 1
        self.num_bins = num_bins
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        sample_points = torch.linspace(0, 1, steps=num_bins, dtype=torch.float32, requires_grad=False)
        self.register_buffer('sample_points', sample_points)

    def forward(self, data):
        batch_size, num_instances, num_features = data.size()

        sample_points = self.sample_points.repeat(batch_size, num_instances, num_features, 1)
        # sample_points.size() --> (batch_size,num_instances,num_features,num_bins)

        data = torch.reshape(data, (batch_size, num_instances, num_features, 1))
        # models.size() --> (batch_size,num_instances,num_features,1)

        diff = sample_points - data.repeat(1, 1, 1, self.num_bins)
        diff_2 = diff ** 2
        # diff_2.size() --> (batch_size,num_instances,num_features,num_bins)

        beta = self.beta or (1. / num_instances)
        result = self.alpha * torch.exp(beta * diff_2)
        # result.size() --> (batch_size,num_instances,num_features,num_bins)

        out_unnormalized = torch.sum(result, dim=1)
        # out_unnormalized.size() --> (batch_size,num_features,num_bins)

        norm_coeff = torch.sum(out_unnormalized, dim=2, keepdim=True)
        # norm_coeff.size() --> (batch_size,num_features,1)

        out = out_unnormalized / norm_coeff
        # out.size() --> (batch_size,num_features,num_bins)

        return out


class Transformator(Module):
    # Stage 3: Representation Transformator
    # i.e. Multi-layer Perceptron
    def __init__(self, num_features, num_bins, num_classes,
                 num_inner=None):
        super(Transformator, self).__init__()
        num_inner = num_inner or [num_classes * 4, num_classes * 2]
        num_all = [num_features * num_bins, *num_inner, num_classes]
        layers = []
        for m, n in zip(num_all, num_all[1:]):
            if layers:
                layers.append(ReLU())
            layers.append(Dropout(0.5))
            layers.append(Linear(m, n))
        self.fc = Sequential(*layers)

    def forward(self, x):
        x = self.fc(x)
        return x


class Model(Module):
    def __init__(self, num_classes, num_instances, num_features, num_bins, sigma=None):
        super(Model, self).__init__()
        self.num_instances = num_instances
        self.num_features = num_features

        self.extractor = FeatureExtractor(num_features)
        self.pooling_filter = PoolingFilter(num_bins, sigma=sigma)
        self.transformator = Transformator(num_features, num_bins, num_classes)

    def forward(self, x):
        x = self.extractor(x)
        x = torch.reshape(x, (-1, self.num_instances, self.num_features))
        x = self.pooling_filter(x)
        x = torch.flatten(x, 1)
        x = self.transformator(x)
        return x
