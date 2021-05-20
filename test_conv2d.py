import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import data_managers
from shared_utils import run_train_cycle, run_test_cycle, \
    initialize_output_files, write_new_result


class Conv2dBlock(nn.Module):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_shape[0], out_shape, (3, 3), (1, 1)),
            nn.BatchNorm2d(out_shape),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.module(x)


class PoolingConv2dBlock(nn.Module):
    def __init__(self, in_shape, out_shape, drop_rate):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_shape[0], out_shape, (3, 3), (1, 1)),
            nn.BatchNorm2d(out_shape),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return self.module(x)


class LinearBlock(nn.Module):
    def __init__(self, in_shape, out_shape, drop_rate):
        super().__init__()
        self.module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(in_shape), out_shape),
            nn.BatchNorm1d(out_shape),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate)
        )

    def forward(self, x):
        return self.module(x)


class Conv2dNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Build up the model
        self.conv1 = Conv2dBlock((3, 32, 32), 32)
        self.conv2 = PoolingConv2dBlock((32, 30, 30), 32, 0.25)
        self.conv3 = Conv2dBlock((32, 14, 14), 64)
        self.conv4 = PoolingConv2dBlock((64, 12, 12), 64, 0.25)
        self.linear = LinearBlock((64, 5, 5), 512, 0.5)
        self.classif = nn.Sequential(
            nn.Linear(512, 20),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.linear(x)
        return self.classif(x)


if __name__ == '__main__':
    # Determine where to save the results
    parser = argparse.ArgumentParser(
        description="Run the progressive ConvNet")

    parser.add_argument('-o', '--output',
                        help="The destination the results of this test "
                             "should be placed",
                        required=True)

    args = parser.parse_args()

    out_path = Path(args.output)

    # Initialize the output files for the results of this test
    initialize_output_files(out_path)

    # Lock in the seed so our experiments are replicable
    torch.manual_seed(36246)

    # Identify whether a CUDA GPU is available to run the analyses on
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # The parameters to run the model with
    no_cycles = 10
    train_epochs = 90
    prune_epochs = 10
    in_shape = (3, 32, 32)
    train_lr = 0.001
    batch_size = 256

    training_data = CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=data_managers.train_transform
    )

    testing_data = CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=data_managers.test_transform
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

    # Build the loss function for the training
    loss_fn = nn.CrossEntropyLoss()

    for c in range(no_cycles):
        # Build the model, initialized with one column
        model = Conv2dNet().to(device)

        optim = torch.optim.Adam(model.parameters(), lr=train_lr,
                                 betas=(0.99, 0.999), weight_decay=0.001)
        # Train the model, tracking the results
        for e in range(train_epochs):
            print(f"=============Cycle {c} Epoch {e}=============")
            train_result = run_train_cycle(model, train_dataloader, loss_fn,
                                           optim, device)
            write_new_result(c, e, train_result['accuracy'],
                             train_result['loss'], out_path, is_train=True)

            test_result = run_test_cycle(model, testing_dataloader, loss_fn, device)
            write_new_result(c, e, test_result['accuracy'],
                             test_result['loss'], out_path, is_train=False)