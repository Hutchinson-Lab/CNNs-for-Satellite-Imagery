import numpy as np
from time import time
import pandas
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.nn as nn
from src.datasets import satellite_dataloader
import sys
from time import time



def train_model(model, cuda, dataloader, optimizer, epoch, criterion, num_classes, model_type, print_every=100):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    
    model.train()
    t0 = time()
    sum_loss = 0
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0
    idx = 0

    for img, label, _ in dataloader:
        if cuda:
            img = img.cuda()
            label = label.cuda()

        optimizer.zero_grad()

        outputs = torch.squeeze(model(img)).to(torch.float64)
        if num_classes == 1:
            outputs = outputs.double()
            label = label.double()

        loss = criterion(outputs, label)

        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (
                print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
        idx += 1
    avg_loss = sum_loss / n_batches
    print('\nTrain Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    return avg_loss


def validate_model(model, cuda, dataloader, epoch, criterion, num_classes, timeit=False):
    """
    Validates model using the provided dataloader.
    """

    with torch.no_grad():   
        model.eval()
        t0 = time()
        sum_loss = 0
        n_train, n_batches = len(dataloader.dataset), len(dataloader)

        for img, label, _ in dataloader:
            if cuda:
                img = img.cuda()
                label = label.cuda()

            if timeit:
                t_start = time()
                outputs = torch.squeeze(model(img)).to(torch.float64)
                t_end = time()
            else:
                outputs = torch.squeeze(model(img)).to(torch.float64)
            if num_classes == 1:
                outputs = outputs.double()
                label = label.double()

            loss = criterion(outputs, label)

            sum_loss += loss.item()
        avg_loss = sum_loss / n_batches
        print('Test Epoch {}: Loss {:0.4f}, Time {:0.3f}s'.format(epoch, avg_loss, time()-t0))
    if timeit:
        return avg_loss, t_end-t_start
    else:
        return avg_loss
