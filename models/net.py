import torch
import torch.nn as nn
from models.efficientnet_pytorch.model import EfficientNet
from models.modelzoo.senet2 import seresnet34
from models.modelzoo.inceptionresnetv2 import inceptionresnetv2
from models.modelzoo.inceptionV4 import inceptionv4
from torchvision.models.densenet import *
from models.modelzoo.senet2 import seresnext50_32x4d
from models.resnest.resnest import resnest50
from collections import OrderedDict
from torch.utils import model_zoo
from torch.nn import functional as F
import math

url_path = {'kaggle':{'efficientnet-b0':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b0-b64d5a18.pth',
                      'efficientnet-b1':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b1-0f3ce85a.pth',
                      'efficientnet-b2':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b2-6e9d97e5.pth',
                      'efficientnet-b3':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b3-cdd7c0f4.pth',
                      'efficientnet-b4':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b4-44fb3a87.pth',
                      'efficientnet-b5':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b5-86493f6b.pth',
                      'efficientnet-b6':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b6-ac80338e.pth',
                      'efficientnet-b7':'/kaggle/input/efficientnet-pytorch-pretrained/adv-efficientnet-b7-4652b6dd.pth'},
            'github':{'efficientnet-b0': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth',
                      'efficientnet-b1': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth',
                      'efficientnet-b2': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth',
                      'efficientnet-b3': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth',
                      'efficientnet-b4': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth',
                      'efficientnet-b5': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth',
                      'efficientnet-b6': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth',
                      'efficientnet-b7': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth',
                      'efficientnet-b8': 'https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth',}
            }

feature_dict = {
        'efficientnet-b0':  1280, #320,
        'efficientnet-b1':  1280, #320,
        'efficientnet-b2':  1408, #352,
        'efficientnet-b3':  1536, #384,
        'efficientnet-b4':  1792, #448,
        'efficientnet-b5':  2048, #512,
        'efficientnet-b6':  2304, #576,
        'efficientnet-b7':  2560, #640,
}


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        features = features.contiguous().view(features.size(0), -1)
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine

class efficientnet(nn.Module):
    def __init__(self, net_name, num_classes=5, weight_path='github'):
        super(efficientnet, self).__init__()
        # self.model = EfficientNet.from_pretrained(size_dict[size])
        # self.feature_size = feature_dict[size]
        state = torch.load(url_path[weight_path][net_name])
        model = EfficientNet.from_name(net_name)
        model.load_state_dict(state)
        self.model = model
        self.feature_size = feature_dict[net_name]

        self.dropout = nn.Dropout(0.2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(self.feature_size, num_classes)
        self.arc = ArcMarginProduct(self.feature_size, num_classes)

    def logits(self, x):
        x = self.avg_pool(x)
        arc = self.arc(x)
        x = self.dropout(x)
        x = x.contiguous().view(x.size(0), -1)
        x = self.last_linear(x)
        return x, arc

    def forward(self, x):

        x = self.model.extract_features(x)
        x, arc = self.logits(x)
        return x, arc

    def get_features(self):
        return self.model.extract_features

    def re_init(self):
        self.last_linear.weight.data.normal_(0, 0.01)
        self.last_linear.bias.data.zero_()



def get_net(modelName, num_classes, weight_path):
    if modelName == 'efficientnet-b0':
        print('using efficientnet-b0')
        state = model_zoo.load_url(url_path[weight_path]['efficientnet-b0'])
        model = EfficientNet.from_name("efficientnet-b0")
        model.load_state_dict(state)
        model._fc = torch.nn.Linear(1280, num_classes)
        return model
    elif modelName == 'efficientnet-b2':
        print('using efficientnet-b2')
        state = model_zoo.load_url(url_path[weight_path]['efficientnet-b2'])
        model = EfficientNet.from_name("efficientnet-b2")
        model.load_state_dict(state)
        # model._avg_pooling = GeM()
        model._fc = nn.Sequential(
            torch.nn.Linear(1408, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model
    elif modelName == 'efficientnet-b3':
        print('using efficientnet-b3')
        # 必须使用该方法下载模型，然后加载
        state = model_zoo.load_url(url_path[weight_path]['efficientnet-b3'])
        model = EfficientNet.from_name("efficientnet-b3")
        model.load_state_dict(state)
        model._fc = nn.Sequential(
            torch.nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model
    elif modelName == 'efficientnet-b4':
        print('using efficientnet-b4')
        # 必须使用该方法下载模型，然后加载
        state = model_zoo.load_url(url_path[weight_path]['efficientnet-b4'])
        model = EfficientNet.from_name("efficientnet-b4")
        model.load_state_dict(state)
        # model._avg_pooling = GeM()
        model._fc = nn.Sequential(
            torch.nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model

    elif modelName == 'efficientnet-b5':
        print('using efficientnet-b5')
        # 必须使用该方法下载模型，然后加载
        state = model_zoo.load_url(url_path[weight_path]['efficientnet-b5'])
        model = EfficientNet.from_name("efficientnet-b5")
        model.load_state_dict(state)
        # model._avg_pooling = GeM()
        model._fc = nn.Sequential(
            torch.nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model

    elif modelName == 'efficientnet-b6':
        print('using efficientnet-b6')
        # 必须使用该方法下载模型，然后加载
        state = model_zoo.load_url(url_path[weight_path]['efficientnet-b6'])
        model = EfficientNet.from_name("efficientnet-b6")
        model.load_state_dict(state)
        # model._avg_pooling = GeM()
        model._fc = nn.Sequential(
            torch.nn.Linear(2304, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model

    elif modelName == 'efficientnet-b7':
        print('using efficientnet-b7')
        # 必须使用该方法下载模型，然后加载
        state = model_zoo.load_url(url_path[weight_path]['efficientnet-b7'])
        model = EfficientNet.from_name("efficientnet-b7")
        model.load_state_dict(state)
        # model._avg_pooling = GeM()
        model._fc = nn.Sequential(
            torch.nn.Linear(2560, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        return model

    elif modelName == 'senet34':
        print('using senet34')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/seresnet34-a4004e63.pth')
        model = seresnet34()
        model.load_state_dict(torch.load(path))
        model.last_linear = torch.nn.Linear(512, num_classes)
        return model
    elif modelName == 'inceptionresnetv2':
        print('using inceptionresnetv2')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/inceptionresnetv2-520b38e4.pth')
        model = inceptionresnetv2(pretrained=False)
        model.load_state_dict(torch.load(path))
        model.last_linear = nn.Sequential(
                torch.nn.Linear(1536, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        return model
    elif modelName == 'inceptionv4':
        print('using inceptionv4')
        path = remote_helper.get_remote_date('https://www.flyai.com/m/inceptionv4-8e4777a0.pth')
        model = inceptionv4(pretrained=False)
        pretrained_dict = torch.load(path)
        model_dict = model.state_dict()
        for k in model_dict.keys():
            if (('module.' + k) in pretrained_dict.keys()):
                model_dict[k] = pretrained_dict.get(('module.' + k))
        model.load_state_dict(model_dict)
        model.last_linear = nn.Sequential(
                torch.nn.Linear(1536, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        return model
    elif modelName == 'seresnext50_32x4d':
        print('using seresnext50_32x4d')
        # 必须使用该方法下载模型，然后加载
        path = remote_helper.get_remote_date('https://www.flyai.com/m/se_resnext50_32x4d-a260b3a4.pth')
        model = seresnext50_32x4d(pretrained=False)
        model.load_state_dict(torch.load(path))
        model.last_linear = nn.Sequential(
                torch.nn.Linear(18432, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
        return model
    elif modelName == 'resnest50':
        print('using resnest50')
        # 必须使用该方法下载模型，然后加载
        path = remote_helper.get_remote_date('https://www.flyai.com/m/resnest50-528c19ca.pth')
        model = resnest50(pretrained=False)
        model.load_state_dict(torch.load(path))
        model.last_linear = nn.Sequential(
                torch.nn.Linear(2048, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        return model
    else:
        print('error,please check your model name!')


if __name__ == '__main__':
    # mobilenet = get_net('efficientnet-b7', 2, weight_path='github')
    # input = torch.rand((1, 3, 300, 300))
    # # print(mobilenet)
    # out = mobilenet(input)
    # print(out.shape)
    name = 'efficientnet-b1'
    num_classes = 5
    model = efficientnet(net_name=name, num_classes=num_classes, weight_path='github')
    import torch

    x = torch.randn([1, 3, 224, 224])
    y_pred, y_metric = model(x)
    pred = F.softmax(y_pred, dim=1).detach().numpy()
    a = pred.argmax(1)
    print(pred)
    print(a)
