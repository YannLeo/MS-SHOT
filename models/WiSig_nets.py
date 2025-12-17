import torch
import torch.nn.utils.weight_norm as weightNorm
from torch import nn

from .resnet1D import ResNet1D



class WiSigEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.feature_extractor = ResNet1D(
            in_channels=2,
            base_filters=64,
            kernel_size=3,
            stride=2,
            n_block=4,
            groups=1,
            n_classes=out_dim,
            downsample_gap=2,
            increasefilter_gap=1,
            verbose=False
        )
    
    def forward(self, x):
        return self.feature_extractor(x)
        

class WiSigNet(WiSigEncoder):
    def __init__(self, num_classes):
        super().__init__(num_classes)



def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)



class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bootleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class WiSigSHOTNet(nn.Module):
    def __init__(self, num_classes, bottleneck_dim=256):
        super().__init__()
        self.netF = WiSigEncoder(1024)
        self.netB = feat_bootleneck(feature_dim=1024, bottleneck_dim=bottleneck_dim, type="bn")
        self.netC = feat_classifier(class_num=num_classes, bottleneck_dim=bottleneck_dim, type="wn")

    def forward(self, x, return_feature=False):
        feature = self.netB(self.netF(x))
        # 2023.4.6: add norm
        # feature = feature / torch.norm(feature, dim=1, p=2, keepdim=True)
        # feature = nn.functional.normalize(feature, dim=1) 
        #
        out = self.netC(feature)
        if return_feature:
            return out, feature
        return out
