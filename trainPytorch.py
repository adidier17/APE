"""
reference: https://github.com/fastai/fastai_old/blob/master/dev_nb/001a_nn_basics.ipynb
"""

import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.utils.data import TensorDataset
import numpy as np


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl: loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb)
                                 for xb, yb in valid_dl])

        val_loss = np.sum(np.multiply(losses,nums)) / np.sum(nums)

        print(epoch, val_loss)

def get_data(x, y, val_idx, bs):
    train_idx = list(set(range(x.shape[0])) - set(val_idx))
    x_val = x[val_idx]
    x_train = x[train_idx]
    print(f"xtrain {x_train}")
    y = np.asarray(y, dtype=float)
    y_val = y[val_idx]
    y_train = y[train_idx]
    print(f"ytrain {y_train}. dtype: {y_train.dtype}")
    x_train, x_val = map(torch.tensor, [x_train, x_val])
    y_train, y_val = map(lambda x: torch.tensor(x, dtype=torch.float), [y_train,  y_val])
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_val, y_val)

    return (DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(valid_ds, batch_size=bs * 2))