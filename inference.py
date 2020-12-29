import pandas as pd
import torch
from torch.utils.data import DataLoader
from models.net import efficientnet
import logging
import sys, os
from config import Config
import warnings
from dataset.dataset import FGVC7Data
from utils.utils import get_transform
from tqdm import tqdm
from torch.nn import functional as F
# parser = argparse.ArgumentParser()
# parser.add_argument('--datasets', default='./data/', help='Train Dataset directory path')
# parser.add_argument('--net', default='b4', help='Choose net to use')
# args = parser.parse_args()
config = Config()
# config.net = args.net
# config.refresh()

# GPU settings
assert torch.cuda.is_available()
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True
num_classes = 5
# def choose_net(name: str):
#     if len(name) == 2 and name[0] == 'b':
#         model = efficientnet(size=name, num_classes=num_classes)
#     elif name.lower() == 'seresnext50':
#         model = se_resnext50(num_classes=num_classes)
#     elif name.lower() == 'seresnext101':
#         model = se_resnext101(num_classes=num_classes)
#     elif name.lower() == 'resnest101':
#         model = Resnest101(num_classes=num_classes)
#     elif name.lower() == 'resnest200':
#         model = Resnest200(num_classes=num_classes)
#     elif name.lower() == 'resnest269':
#         model = Resnest269(num_classes=num_classes)
#     elif name.lower() == 'densenet121':
#         model = DenseNet121(num_classes=num_classes)
#     else:
#         logging.fatal("The net type is wrong.")
#         sys.exit(1)
#     return model

datasets = '/kaggle/input/cassava-leaf-disease-classification'

batch_size, num_workers = 1, 4
net_name = 'efficientnet-b2'
def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    try:
        ckpt = '../input/efefb2b2/weights/model_b2_386.ckpt'
    except:
        print('Set ckpt for evaluation in config.py')
        return

    test_dataset = FGVC7Data(root=datasets, phase='test',
                             transform=get_transform([config.image_size[0], config.image_size[1]], 'test'))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True)
    num_classes = 5
    net = efficientnet(net_name=net_name, num_classes=num_classes, weight_path='kaggle')
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    print('Network loaded from {}'.format(ckpt))
    net = net.to(device)
    preds = []
    image_name = []
    with torch.no_grad():
        net.eval()
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        for i, input in enumerate(test_loader):
            X, _, img_name = input
            y_pred, y_metric  = net(X.to(device))
            pred = F.softmax(y_pred, dim=1).cpu().numpy()
            preds.extend(pred.argmax(1))
            image_name.append(img_name[0])
    sub = pd.DataFrame({'image_id': image_name, 'label': preds})
    print(sub)
    sub.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()
