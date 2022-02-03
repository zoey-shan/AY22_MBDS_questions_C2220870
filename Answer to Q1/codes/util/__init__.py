import os
import time
from datetime import datetime

import torch
from torch.nn import MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from util.logs import logger


DATA_ROOT = 'cache/'
MODELS_DIR = 'models/'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def target_transformer(target):
    mapper = [0, 7, 2, 3, 4, 5, 6, 1, 8, 9]
    return mapper[target]


def get_data_loader(batch_size, train=True):
    if not os.path.isdir(DATA_ROOT):
        os.makedirs(DATA_ROOT)
    dataset = MNIST(root=DATA_ROOT, download=True, train=train, transform=ToTensor(),
                    target_transform=target_transformer)
    idx = dataset.targets <= 1
    dataset.data, dataset.targets = dataset.data[idx], dataset.targets[idx]
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    return loader


def train(model, num_epochs, batch_size=None, device=None):
    # batch_size = batch_size or model.num_instances
    iter_cnt = batch_size // model.num_instances
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.isdir(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    data_loader = get_data_loader(batch_size=batch_size, train=True)
    model.to(device)
    optimizer = Adam(model.parameters())
    criterion = MSELoss()

    logger.info(f'-- Training --')
    train_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    min_loss = None
    min_filename = ''
    start_time = time.time()
    model.train()
    loss_ls = []
    acc_ls = []
    for epoch_no in range(num_epochs):
        logger.info(f'Epoch #{epoch_no}')
        total_loss = 0
        total_acc = 0
        cnt = 0
        for data, label in data_loader:
            if data.size(0) < batch_size:
                logger.debug(f'Skipping data w/ size: {data.size()}')
                continue
            data = data.to(device)
            label = label.to(device)

            label = torch.FloatTensor([torch.count_nonzero(label[i:i + model.num_instances] == 0) / model.num_instances
                                       for i in range(0, batch_size, model.num_instances)]).to(device)
            label = torch.unsqueeze(label, dim=1)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            total_loss += loss.sum()
            cnt += iter_cnt
            accuracy = 1 - (torch.abs(output - label) / label)
            total_acc += accuracy.sum()
            logger.debug(f'Label: {label}')
            logger.debug(f'Output: {output}')
            # logger.debug(f'Loss: {loss}')
            loss.backward()
            optimizer.step()

        model_filename = f'model_weights__{train_timestamp}__Ep{epoch_no:05d}.pth'
        logger.debug(f'{total_loss}, {cnt}')
        epoch_loss = total_loss / cnt
        loss_ls.append(epoch_loss)
        epoch_acc = total_acc / cnt
        acc_ls.append(epoch_acc)
        if min_loss is None or min_loss > epoch_loss:
            min_filename = model_filename
            min_loss = epoch_loss
        logger.info(f'End of Epoch #{epoch_no}: Loss={epoch_loss}')
        state_dict = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(state_dict, os.path.join(MODELS_DIR, model_filename))

    end_time = time.time()
    logger.info(f'-- End of training: {end_time - start_time:.2f} second(s) elapsed --')
    logger.info(f'Minimum loss: {min_loss}, corresponding state stored in {min_filename}')
    plt.figure()
    plt.plot(range(num_epochs), loss_ls)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.savefig('loss_tendencyL2.png')
    plt.figure()
    plt.plot(range(num_epochs), acc_ls)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.savefig('acc_tendencyL2.png')
    return min_filename


def test(model, state_filename, batch_size=None, device=None):
    batch_size = batch_size or model.num_instances
    iter_cnt = batch_size // model.num_instances
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = get_data_loader(batch_size=batch_size, train=False)
    state_dict = torch.load(os.path.join(MODELS_DIR, state_filename), map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])

    logger.info(f'-- Testing --')
    model.eval()
    with torch.no_grad():
        total_acc = 0
        cnt = 0
        for data, label in data_loader:
            if data.size(0) < batch_size:
                logger.debug(f'Skipping data w/ size: {data.size()}')
                continue
            data = data.to(device)
            label = label.to(device)

            label = torch.FloatTensor([torch.count_nonzero(label[i:i + model.num_instances] == 0) / model.num_instances
                                       for i in range(0, batch_size, model.num_instances)]).to(device)
            label = torch.unsqueeze(label, dim=1)
            output = model(data)
            accuracy = 1 - (torch.abs(output - label) / label)
            total_acc += accuracy.sum()
            cnt += iter_cnt

        acc = total_acc / cnt
        logger.info(f'Test accuracy: {acc:.2f}')
    logger.info(f'-- End of testing --')
