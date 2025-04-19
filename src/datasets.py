from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import glob
import os
import numpy as np
import random
import pandas as pd


class SatelliteDataset(Dataset):

    def __init__(self, img_dir, label_fn, split, size, img_ext, bands, transform=None):
        self.img_dir = img_dir
        self.size = size
        self.img_files = glob.glob(os.path.join(self.img_dir, '*'))
        self.transform = transform
        self.img_ext = img_ext
        self.bands = bands
        self.label_fn = label_fn
        self.ids, self.labels = get_ids_and_labels_from_npy(split, label_fn)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        single_id = self.ids[idx]
        single_label = self.labels[idx]  # type float

        try:
            if self.img_ext == 'png':
                img = imageio.imread(os.path.join(self.img_dir, single_id + '.png'))
                img = np.asarray(img)  # shape: (X, Y, channels)
            elif self.img_ext == 'jpg':
                img = imageio.imread(os.path.join(self.img_dir, single_id + '.jpg'))
                img = np.asarray(img)  # shape: (X, Y, channels)
            elif self.img_ext == 'npy':
                img = np.load(os.path.join(self.img_dir, single_id + '.npy'))

            if self.bands == 1:  # (X, Y) --> (1, X, Y)
                img = np.expand_dims(img, axis=0)
            elif self.bands == -1:  # ResNet: copy grayscale image to all three channels
                img = np.array([img] * 3)  # (X, y) --> (3, X, Y)
            else:  # reshape: (X, Y, channel) --> (channel, X, Y)
                img = np.moveaxis(img, -1, 0)
            img = img[:, 0:self.size, 0:self.size]  # [channel, w, h]

            if self.transform:
                img = self.transform(img)

            return img, single_label, single_id

        except Exception as e:
            raise Exception(f'Could not open {single_id}')
            print(e)


class GetBands(object):
    """
    Gets the first X bands of the tile triplet.
    """

    def __init__(self, bands):
        assert bands >= 0, 'Must get at least 1 band'
        self.bands = bands

    def __call__(self, img):
        # Tiles are already in [c, w, h] order
        img = img[:self.bands, :, :]
        return img


class ClipAndScale(object):
    """
    Clips and scales bands to between [0, 1] for NAIP, RGB, and Landsat
    satellite images. Clipping applies for Landsat only.
    """

    def __init__(self, img_type):
        assert img_type in ['rgb', 'landsat', 'sentinel-2']
        self.img_type = img_type

    def __call__(self, img):
        if self.img_type in ['rgb', 'sentinel-2']:  # eurosat (sentinel-2) data previously scaled 0-255
            return img / 255

        elif self.img_type == 'landsat':
            return np.clip(img, 0, 255) / (255)


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """

    def __call__(self, img):
        img = torch.from_numpy(img).float()
        return img


class RandomFlipAndRotate(object):
    """
    Does data augmentation during training by randomly flipping (horizontal
    and vertical) and randomly rotating (0, 90, 180, 270 degrees). Keep in mind
    that pytorch samples are CxWxH.
    """

    def __call__(self, img):
        # Randomly horizontal flip
        if np.random.rand() < 0.5: img = np.flip(img, axis=2).copy()
        # Randomly vertical flip
        if np.random.rand() < 0.5: img = np.flip(img, axis=1).copy()
        # Randomly rotate
        rotations = np.random.choice([0, 1, 2, 3])
        if rotations > 0: img = np.rot90(img, k=rotations, axes=(1, 2)).copy()
        return img


def satellite_dataloader(img_type, img_dir, label_fn, split, size, img_ext, bands,
                         batch_size=4, shuffle=True, num_workers=4, means=None):
    """
    Returns a DataLoader with either NAIP (RGB/IR), RGB, or Landsat tiles.
    Turn shuffle to False for producing embeddings that correspond to original
    tiles.
    """
    transform_list = [GetBands(bands), RandomFlipAndRotate(), ToFloatTensor(), ClipAndScale(img_type)]
    if means is not None:
        transform_list.append(transforms.Normalize(means, (1,) * bands))  # do not scale by standard deviation

    transform = transforms.Compose(transform_list)

    dataset = SatelliteDataset(img_dir, label_fn, split, size, img_ext, bands, transform=transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=False)
    return dataloader


def get_ids_and_labels_from_npy(split, label_fn):
    if 'elevation' in label_fn or 'treecover' in label_fn or 'nightlights' in label_fn or 'population' in label_fn:
        label_col = 'label_normalized'
    else:
        label_col = 'label'
    
    label_df = pd.read_csv(label_fn)


    if split == 'all':
        ids = label_df['id'].tolist()  # convert column to list
        labels = label_df[label_col].tolist()
    else:
        ids = label_df.loc[label_df['fold'] == split, ['id']]
        ids = ids['id'].tolist()
        labels = label_df.loc[label_df['fold'] == split, label_col]
        labels = labels.tolist()

    print(f'split: {split}')
    print(f'len ids: {str(len(ids))}')
    return ids, labels
