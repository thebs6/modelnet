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


def get_model_imgs():
    random_img_path = r'D:\Study\ml\test\rotation_dataset'
    imgs = os.listdir(random_img_path)
    imgs_dicts = []
    for idx, img_folder in enumerate(imgs):
        label = idx
        c = img_folder
        imgs_v = []
        for img in os.listdir(os.path.join(random_img_path, img_folder)):
            im = Image.open(os.path.join(random_img_path, img_folder, img))
            im = tf.Resize([224, 224])(im)
            im = np.expand_dims(im, axis=0)
            imgs_v.append(im)
        imgs_v = np.concatenate(imgs_v, axis=0)
        imgs_dicts.append({
            'class': c,
            'label': idx,
            'imgs_v': imgs_v
        })
    return imgs_dicts


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder',
                        default=r'D:\Study\fly\experiments\VLP2_3000')
    parser.add_argument('--init_csv', default=r'D:\Study\fly\paper_code\LearningToCompare_FSL\demo\vlp3.csv')
    parser.add_argument('--model_dict', default='./model_dict.pkl')
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--t_batch', default=16, type=int)
    parser.add_argument('--v_batch', default=64, type=int)
    parser.add_argument('--image_size', default=224, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--model_folder', default='./models')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--scheduler', default='StepLR')
    parser.add_argument('--step_size', default=15, type=int)
    parser.add_argument('--T_max', default=20)
    parser.add_argument('--model', default='resnet18')
    parser.add_argument('--pretrain', default=True)
    parser.add_argument('--optimizer', default='SGD')
    parser.add_argument('--freeze', default=True)
    parser.add_argument('--model_mode', default=0, type=int)
    parser.add_argument('--account', type=str)
    parser.add_argument('--image_folder_type', default=1, type=int)
    parser.add_argument('--model_img_per_class', default=5, type=int)
    parser.add_argument('--d3model_pretrain', default=True, type=bool)
    parser.add_argument('--valid_mode', default=0, type=int)
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


def train_epoch(epo, train_loader, model_imgs, model, optimizer, loss_fn, model_dict, model_img_per_class):
    model.train()
    epoch_loss = 0.0
    accuracy = 0.0
    sb = 0.0
    bar = tqdm(train_loader, total=len(train_loader), position=0)
    for batch, target in bar:
        batch, target = batch.cuda(), target.cuda()
        model_images = torch.from_numpy(model_imgs(model_dict, model_img_per_class, 0)).to(torch.float32).cuda()
        output = model(model_images, batch)
        loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        accuracy += (output.argmax(1) == target).sum()
        optimizer.step()
        epoch_loss += loss.item()
        sb += batch.shape[0]
        bar.set_postfix({'epoch': epo, "train_acc": (accuracy / sb).item(), "train_loss": epoch_loss / sb})
    return accuracy / (train_loader.batch_size * len(train_loader)), epoch_loss / len(train_loader)


@torch.no_grad()
def valid_epoch(epo, valid_loader, model_imgs, model, loss_fn, model_dict, model_img_per_class):
    model.eval()
    epoch_loss = 0.0
    accuracy = 0.0
    bar = tqdm(valid_loader, total=len(valid_loader), position=0)
    sb = 0
    for batch, target in bar:
        batch, target = batch.cuda(), target.cuda()
        model_images = torch.from_numpy(model_imgs(model_dict, model_img_per_class, 0)).to(torch.float32).cuda()
        output = model(model_images, batch)
        loss = loss_fn(output, target)
        accuracy += (output.argmax(1) == target).sum()
        epoch_loss += loss.item()
        sb += batch.shape[0]
        bar.set_postfix({'epoch': epo})
        bar.set_postfix({"valid_acc": (accuracy / sb).item(), "valid_loss": epoch_loss / sb})
    return accuracy / (valid_loader.batch_size * len(valid_loader)), epoch_loss / len(valid_loader)


def get_model_img(model_dict, model_img_per_class, item):
    model_img = [v['imgs_v'][random.randint(0, 100, model_img_per_class)] for v in model_dict]
    model_img = np.concatenate(np.expand_dims(model_img, axis=0), axis=0)
    model_img = np.transpose(model_img, (0, 4, 1, 2, 3))
    return model_img


def get_valid_loader(valid_mode):
    # 0: ????????????
    # Todo 1: ??????????????????????????????10???
    if valid_mode == 0:
        pass



if __name__ == '__main__':
    strtime = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    print("start_time:", strtime)
    args = parse_opt()
    print(args)
    log_path = args.log_path + '/' + strtime
    writer = SummaryWriter(log_path)

    train_transform = tf.Compose([
        tf.RandomResizedCrop(args.image_size),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transform = tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # model_dict = get_model_imgs()[0:-1]
    with open(args.model_dict, 'rb') as tf:
        model_dict = pickle.load(tf)
    # model_v = [np.expand_dims(v['imgs_v'], axis=0) for v in model_dict]
    # model_v = np.concatenate(model_v, axis=0)
    learn_rate = args.lr
    model_img_per_class = args.model_img_per_class

    # ----------------------- ??????-----------------------#
    feature_size = 256
    model_encoder = models.video.r3d_18(args.d3model_pretrain).cuda()
    model_encoder.fc = nn.Linear(model_encoder.fc.in_features, feature_size).cuda()
    img_encoder = get_model_encoder(feature_size, args.pretrain).cuda()
    relation_net = nn.Sequential(
        nn.Linear(feature_size * 2 * 9, feature_size // 2),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(feature_size // 2, feature_size // 4),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(feature_size // 4, 9),
        nn.LogSoftmax(dim=1)
    ).cuda()
    model = net(model_encoder, img_encoder, relation_net)
    # ----------------------- ??????-----------------------#

    # ----------------------- ????????????-----------------------#
    model_optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    model_scheduler = StepLR(model_optimizer, step_size=10, gamma=0.1)
    # ----------------------- ????????????-----------------------#

    # ----------------------- ????????????-----------------------#
    train_data = PDataSet(args, args.init_csv, 'train', transform=train_transform, model_dict=model_dict,
                          model_img_per_class=model_img_per_class)
    imgs_loader = DataLoader(train_data, batch_size=args.t_batch, shuffle=True, num_workers=args.workers,
                             drop_last=False)

    valid_data = PDataSet(args, args.init_csv, 'valid', transform=train_transform, model_dict=model_dict,
                          model_img_per_class=model_img_per_class)
    valid_loader = DataLoader(valid_data, batch_size=args.v_batch, shuffle=True, num_workers=args.workers,
                              drop_last=False)
    # ----------------------- ????????????-----------------------#

    # ----------------------- ??????-----------------------#
    epoch = args.epoch
    loss_fn = nn.CrossEntropyLoss()
    min_loss = 1.0
    for epo in range(1, 1 + epoch):
        train_acc, train_loss = train_epoch(epo, imgs_loader, get_model_img, model, model_optimizer, loss_fn,
                                            model_dict=model_dict, model_img_per_class=model_img_per_class)
        valid_acc, valid_loss = valid_epoch(epo, valid_loader, get_model_img, model, loss_fn,
                                            model_dict=model_dict, model_img_per_class=model_img_per_class)

        writer.add_scalars("loss", {"train_loss": train_loss, "valid_loss": valid_loss}, epo)
        writer.add_scalars("accuracy", {"train_acc": train_acc, "valid_acc": valid_acc}, epo)
        writer.add_scalar("lr", model_optimizer.state_dict()['param_groups'][0]['lr'])

        print('train_loss', train_loss, 'train_acc', train_acc.item())
        print('valid_loss', valid_loss, 'valid_acc', valid_acc.item())
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), f"{args.model_folder}/model{epo}_{train_loss}.pth")
        torch.save(model.state_dict(), f"{args.model_folder}/last_model.pth")
        model_scheduler.step()
    # ----------------------- ??????-----------------------#



