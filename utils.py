import numpy as np
import os
import glob
import random
from time import time
import src.data_utils
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
from time import sleep
from skimage import io
from skimage.feature import local_binary_pattern
import sklearn.metrics as metrics
import src.datasets
import pandas as pd
import cv2


def get_channel_mean_stdDev (dataloader, bands):
    means, stdDevs = None, None
    names = []
    for img, label, id in dataloader:
        img = img.reshape(img.shape[0], bands, -1)  # flatten img w & h (maintain batch size and channels)
        if means is None:
            means = torch.mean(img, dim=(0, 2))  # reduce over batch size and img pixels (take means over channels)
        else:
            means += torch.mean(img, dim=(0, 2))
        if stdDevs is None:
            stdDevs = torch.std(img, dim=(0, 2))  # reduce over batch size and img pixels (take means over channels)
        else:
            stdDevs += torch.std(img, dim=(0, 2))
        names.append(id)
    means = means / len(dataloader)
    stdDevs = stdDevs / len(dataloader)
    return means, stdDevs


def tif_to_npy():
    count = 0

    for img in os.listdir(paths.tif_dir):
        img_name = img.split('.tif')[0]
        print(img_name)
        if img_type == 'nlcd':
            img = load_nlcd(paths.tif_dir + img)
        elif img_type == 'rgb':
            img = load_rgb(paths.tif_dir + img, bands, bands_only=True, is_npy=False)
        else:
            img = load_tif(paths.tif_dir + img, bands, bands_only=True, is_npy=False)
        if (img.shape[0] >= 48 and img.shape[1] >= 48):
            np.save(os.path.join(paths.npy_dir, img_name + '.npy'), img)

        count += 1
    print('Saved ' + str(count) + ' images to ' + paths.npy_dir)


# based on https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


### Evalution metrics ###
def get_model_outputs(CNN, dataloader, cuda):
    CNN.eval()
    with torch.no_grad():
        all_predictions = None
        for imgs, labels, ids in dataloader:
            if cuda:
                imgs = imgs.to('cuda')

            if all_predictions is None:
                all_predictions = CNN(imgs)
                all_labels = labels
                all_ids = ids
            else:
                model_outputs = CNN(imgs)
                all_predictions = torch.cat([all_predictions, model_outputs], dim=0)
                all_labels = torch.cat([all_labels, labels], dim=0)
                all_ids += ids

    return all_predictions, all_labels, all_ids


def calc_r2(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()

    r2 = metrics.r2_score(labels, predictions)  # y_true, y_pred
    return r2


def calc_mse(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()
    mse = metrics.mean_squared_error(labels, predictions)
    return mse


def calc_mae(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    predictions = predictions.detach().cpu().numpy().ravel()
    labels = labels.detach().cpu().numpy()
    mae = metrics.mean_absolute_error(labels, predictions)
    return mae


def calc_acc(CNN, dataloader, cuda, num_classes):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()

    # convert raw model outputs to classes
    if num_classes == 1:
        m = nn.Sigmoid()  # nn.Softmax()
        sig_preds = m(predictions).detach().cpu().numpy()
        pred_classes = [0 if x < 0.5 else 1 for x in sig_preds]  # o.5 only becasue coffee is balanced

    else:
        m = nn.Softmax(dim=1)
        sf_max_preds = m(predictions)
        pred_classes = torch.argmax(sf_max_preds, dim=1)
        pred_classes = pred_classes.detach().cpu().numpy()

    # calc accuracy
    acc = metrics.accuracy_score(labels, pred_classes)
    print(acc)
    return acc


def calc_confusion_matrix(CNN, dataloader, cuda):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()


    m = nn.Softmax(dim=1)
    sf_max_preds = m(predictions)
    pred_classes = torch.argmax(sf_max_preds, dim=1)
    pred_classes = pred_classes.detach().cpu().numpy()

    # calc accuracy
    confusion = metrics.confusion_matrix(labels, pred_classes)
    return confusion


def calc_PR(CNN, dataloader, cuda, num_classes):
    predictions, labels, _ = get_model_outputs(CNN, dataloader, cuda)
    labels = labels.detach().cpu().numpy()

    # convert raw model outputs to classes
    if num_classes == 1:
        m = nn.Sigmoid()  # nn.Softmax()
        sig_preds = m(predictions).detach().cpu().numpy()
        pred_classes = [0 if x < 0.5 else 1 for x in sig_preds]  # o.5 only becasue coffee is balanced
    else:
        m = nn.Softmax(dim=1)
        sf_max_preds = m(predictions)
        pred_classes = torch.argmax(sf_max_preds, dim=1)
        pred_classes = pred_classes.detach().cpu().numpy()

    # confusion matrix
    rep = metrics.classification_report(labels, pred_classes, digits=3, output_dict=True)
    return rep['macro avg'], rep['weighted avg']


def get_predictions(CNN, dataloader, cuda, mode):
    outputs, labels, ids = get_model_outputs(CNN, dataloader, cuda)

    if mode == 'classification':
        # convert raw model outputs to classes
        m = nn.Softmax(dim=1)
        sf_max_preds = m(outputs)
        class_preds = torch.argmax(sf_max_preds, dim=1)
        class_preds = np.expand_dims(class_preds.detach().cpu().numpy(), axis=1)
        sf_max_preds = sf_max_preds.detach().cpu().numpy()

    outputs = outputs.detach().cpu().numpy()
    labels = np.expand_dims(labels.detach().cpu().numpy(), axis=1)
    if mode == 'classification':
        return sf_max_preds, class_preds, labels, np.expand_dims(np.array(ids), axis=1)
    else:
        return outputs, labels, np.expand_dims(np.array(ids), axis=1)
