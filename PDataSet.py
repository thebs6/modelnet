import os

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from numpy import random


class PDataSet(Dataset):
    ## image_folder_type :0 所有图片在一个文件夹
    ## image_folder_type :1 图片按照类别放在不同文件夹
    def __init__(self, args, csv_path, split, transform, model_dict, model_img_per_class, data_type='images'):
        super(PDataSet, self).__init__()
        self.args = args
        self.model_dict = model_dict
        self.model_img_per_class = model_img_per_class
        self.data_type = data_type
        self.df = pd.read_csv(csv_path, dtype={'imageId': str})
        if args.image_folder_type == 0:
            self.df['image_path'] = self.df['imageId'].apply(lambda p: os.path.join(args.image_folder, p + '.jpg'))
        elif args.image_folder_type == 1:
            self.df['image_path'] = self.df.apply(
                lambda p: os.path.join(args.image_folder, p['plane_type'], p['imageId'] + '.jpg'), axis=1)
        self.transform = transform
        self.classes = self.df['plane_type'].unique()
        self.class2label = {c: label for label, c in enumerate(self.classes)}
        self.label2class = {label: c for label, c in enumerate(self.classes)}
        self.df['label'] = self.df['plane_type'].apply(lambda c: self.class2label[c])
        if split not in ['train', 'valid', 'test']:
            raise (ValueError, 'split error')
        self.resdf = self.df[self.df['split'] == split]

    def __len__(self):
        return len(self.resdf)

    def __getitem__(self, item):
        image = Image.open(self.resdf.iloc[item]['image_path'])
        image = image.convert("RGB")
        image = self.transform(image)
        label = self.resdf.iloc[item]['label']
        if self.data_type == 'images':
            return image, label
        # elif self.data_type == 'models':
        #     model_img = [v['imgs_v'][random.randint(0, 100, self.model_img_per_class)] for v in self.model_dict]
        #     model_img = np.concatenate(model_img, axis=0)
        #     return model_img
