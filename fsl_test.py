import argparse
import time

from numpy import random

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as tf
import pandas as pd
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import pickle
from torch.utils.tensorboard import SummaryWriter
from PDataSet import PDataSet


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder',
                        default=r'D:\Study\fly\experiments\crowl_ps_using\images_origin_remove_duplicate')
    parser.add_argument('--init_csv', default=r'vlp_test_JAS39.csv')
    parser.add_argument('--model_dict', default='./model_dict.pkl')
    parser.add_argument('--t_batch', default=16, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--model_path', default='models/model6_0.893963658079809.pth', type=str)
    parser.add_argument('--image_folder_type', default=1, type=int)
    parser.add_argument('--model_img_per_class', default=5, type=int)
    parser.add_argument('--log_path', default='log', type=str)

    args = parser.parse_args()
    return args


def get_model_encoder(feature_size, pretrain):
    net = models.resnet18(pretrained=pretrain)
    net.fc = nn.Linear(net.fc.in_features, feature_size)
    return net


class net(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, model_encoder, img_encoder, relation_net):
        super(net, self).__init__()
        self.model_encoder = model_encoder
        self.img_encoder = img_encoder
        self.relation_net = relation_net

    def forward(self, model_v, img):
        batch = img.shape[0]
        model_feature = self.model_encoder(model_v)  # (9, 256)
        model_feature = model_feature.unsqueeze(0).repeat(batch, 1, 1)  # (batch, 9, 256)
        img_feature = self.img_encoder(img)  # (batch, 256)
        img_feature = img_feature.unsqueeze(1).repeat(1, 9, 1)  # (batch, 9, 256)
        # similarity = torch.cosine_similarity(img_feature, model_feature, dim=2)
        out = torch.concat([img_feature, model_feature], dim=2)  # (batch, 9, 512)
        out = out.view(batch, -1)
        out = self.relation_net(out)
        return out


@torch.no_grad()
def test_epoch(valid_loader, model_imgs, model, model_dict, model_img_per_class):
    model.eval()
    accuracy = 0.0
    bar = tqdm(valid_loader, total=len(valid_loader), position=0)
    sb = 0
    for batch, target in bar:
        batch, target = batch.cuda(), target.cuda()
        model_images = torch.from_numpy(model_imgs(model_dict, model_img_per_class, 0)).to(torch.float32).cuda()
        output = model(model_images, batch)
        accuracy += (output.argmax(1) == target).sum()
        sb += batch.shape[0]
        bar.set_postfix({"valid_acc": (accuracy / sb).item()})
    return accuracy / (valid_loader.batch_size * len(valid_loader))


def get_model_img(model_dict, model_img_per_class, item):
    model_img = [v['imgs_v'][random.randint(0, 100, model_img_per_class)] for v in model_dict]
    model_img = np.concatenate(np.expand_dims(model_img, axis=0), axis=0)
    model_img = np.transpose(model_img, (0, 4, 1, 2, 3))
    return model_img


if __name__ == '__main__':
    strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    print("start_time:", strtime)
    args = parse_opt()
    print(args)
    log_path = args.log_path + '/' + strtime
    writer = SummaryWriter(log_path)

    test_transform = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with open(args.model_dict, 'rb') as tf:
        model_dict = pickle.load(tf)
    model_img_per_class = args.model_img_per_class

    # ----------------------- 网络-----------------------#
    # feature_size = 256
    # model_encoder = models.video.r3d_18(args.d3model_pretrain).cuda()
    # model_encoder.fc = nn.Linear(model_encoder.fc.in_features, feature_size).cuda()
    # img_encoder = get_model_encoder(feature_size, args.pretrain).cuda()
    # relation_net = nn.Sequential(
    #     nn.Linear(feature_size * 2 * 9, feature_size // 2),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(feature_size // 2, feature_size // 4),
    #     nn.ReLU(),
    #     nn.Dropout(0.4),
    #     nn.Linear(feature_size // 4, 9),
    #     nn.LogSoftmax(dim=1)
    # ).cuda()
    # model = net(model_encoder, img_encoder, relation_net)

    model = torch.load(args.model_path)

    # ----------------------- 网络-----------------------#

    # ----------------------- 数据相关-----------------------#
    test_data = PDataSet(args, args.init_csv, 'test', transform=test_transform, model_dict=model_dict,
                         model_img_per_class=model_img_per_class)
    test_loader = DataLoader(test_data, batch_size=args.t_batch, shuffle=True, num_workers=args.workers,
                             drop_last=False)
    # ----------------------- 数据相关-----------------------#

    # ----------------------- 训练-----------------------#
    test_acc = test_epoch(test_loader, get_model_img, model,
                          model_dict=model_dict, model_img_per_class=model_img_per_class)
    writer.add_scalar("test_acc", test_acc)
    print("test_acc", test_acc)
    # ----------------------- 训练-----------------------#
