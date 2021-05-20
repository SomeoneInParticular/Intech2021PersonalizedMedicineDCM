import argparse
import numpy as np
import pandas as pd
import torch

import data_managers
from data_managers import PairedMetricCIFAR100Coarse
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, random_split

from cifar100coarse.cifar100coarse import CIFAR100Coarse


# Identify the device to use for all processes

device = "cuda" if torch.cuda.is_available() else "cpu"


def conv_chain(in_channel, out_channel) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channel, out_channel, (3, 3), (1, 1)),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
    ]


def pooled_conv_chain(in_channel, out_channel, dropout=0.25) -> list[nn.Module]:
    ret_list = conv_chain(in_channel, out_channel)
    ret_list.append(nn.MaxPool2d((2, 2), (2, 2)))
    ret_list.append(nn.Dropout(dropout))
    return ret_list


class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, no_classes: int):
        super(SimpleCNN, self).__init__()

        self.block1 = nn.Sequential(
            *conv_chain(input_channels, 32)
        )

        self.block2 = nn.Sequential(
            *pooled_conv_chain(32, 32, 0.25)
        )

        self.block3 = nn.Sequential(
            *conv_chain(32, 64)
        )

        self.block4 = nn.Sequential(
            *pooled_conv_chain(64, 64, 0.25)
        )

        self.block5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.interp_block = nn.Linear(512, no_classes)

        self.out_block = nn.Softmax(dim=1)

    def forward(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.interp_block(x)
        out = self.out_block(x)
        return out

    def run_train_cycle(self, dataloader, loss_fn, optim, report_rate=50):

        losses = []

        for i, (X, y) in enumerate(dataloader):
            # Load the data into the device for processing
            X, y = X.to(device), y.to(device)

            # Prediction loss calc
            pred = self(X)
            loss = loss_fn(pred, y)
            losses.append(loss.cpu().detach().numpy())

            # Backpropagation to update the model
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Report loss every 100 batches
            if i % report_rate == 0:
                print(f'Loss: {loss.item()} for batch {i}')

        return np.array(losses)

    def run_test_cycle(self, dataloader, loss_fn):
        # Set this to evaluate mode
        self.eval()
        # Accumulate loss and accuracy across the dataset
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Evaluate the average loss and accuracy across the dataset
        size = len(dataloader.dataset)
        test_loss /= size
        correct /= size
        # Report it to the the console
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # Return a dictionary containing the results (for use in plotting etc.), list wrapped for dataframe usage
        return {
            "loss": [test_loss],
            "accuracy": [correct]
        }


class PairedDenseCNN(nn.Module):
    def __init__(self, image_channels: int, dense_inputs: int, no_classes: int):
        super(PairedDenseCNN, self).__init__()

        # Initialize the CNN branch of the network
        self.cnn_block1 = nn.Sequential(
            *conv_chain(image_channels, 32)
        )

        self.cnn_block2 = nn.Sequential(
            *pooled_conv_chain(32, 32, 0.25)
        )

        self.cnn_block3 = nn.Sequential(
            *conv_chain(32, 64)
        )

        self.cnn_block4 = nn.Sequential(
            *pooled_conv_chain(64, 64, 0.25)
        )

        self.cnn_block5 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        # Initialize the dense branch of the network
        self.dense_branch = nn.Sequential(
            nn.Linear(dense_inputs, dense_inputs),
            nn.Linear(dense_inputs, dense_inputs),
            nn.Linear(dense_inputs, dense_inputs)
        )

        # The "interpretation" block, which attempts to make sense of the two branches combined outputs
        self.interp_block = nn.Linear(512 + dense_inputs, no_classes)

        # The final output block, being a softmax evaluation categorization
        self.out_block = nn.Softmax(dim=1)

    def forward(self, cnn_input, deep_input):
        # Chain the cnn blocks first
        x1 = self.cnn_block1(cnn_input)
        x1 = self.cnn_block2(x1)
        x1 = self.cnn_block3(x1)
        x1 = self.cnn_block4(x1)
        x1 = self.cnn_block5(x1)

        # Chain the deep blocks next
        x2 = self.dense_branch(deep_input)

        # Concatenate the results, and feed it into interp block
        x = torch.cat((x1, x2), dim=1)
        x = self.interp_block(x)

        # Finally, calculate the predicted categories
        cnn_out = self.out_block(x)
        return cnn_out

    def run_train_cycle(self, dataloader, loss_fn, optim, report_rate=50):
        losses = []

        for i, ((X1, X2), y) in enumerate(dataloader):
            # Load the data into the device for processing
            X1, X2, y = X1.to(device), X2.to(device), y.to(device)

            # Prediction loss calc
            pred = self(cnn_input=X1, deep_input=X2)
            loss = loss_fn(pred, y)
            losses.append(loss.cpu().detach().numpy())

            # Backpropagation to update the model
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Report loss every 100 batches
            if i % report_rate == 0:
                print(f'Loss: {loss.item()} for batch {i}')

        return losses

    def run_test_cycle(self, dataloader, loss_fn):
        # Set this to evaluate mode
        self.eval()
        # Accumulate loss and accuracy across the dataset
        test_loss, correct = 0, 0
        with torch.no_grad():
            for (X1, X2), y in dataloader:
                X1, X2, y = X1.to(device), X2.to(device), y.to(device)
                pred = self(cnn_input=X1, deep_input=X2)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        # Evaluate the average loss and accuracy across the dataset
        size = len(dataloader.dataset)
        test_loss /= size
        correct /= size
        # Report it to the the console
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        # Return a dictionary containing the results (for use in plotting etc.)
        return {
            "loss": [test_loss],
            "accuracy": [correct]
        }


if __name__ == '__main__':
    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Run our static networks and record their progress")

    parser.add_argument('-o', '--output', help='The destination in which to place the output files')
    parser.add_argument('-e', '--epochs', help='The number of epochs that should be run on each model per cycle',
                        type=int)
    parser.add_argument('-b', '--batch', help='The size of the batches to submit to the network during training',
                        type=int)

    args = parser.parse_args()

    output_path = args.output

    epochs = args.epochs
    batch_size = args.batch

    del args
    del parser

    # Initialize our dataframe dump files, resetting them if they already exist
    tmp_train_df = pd.DataFrame(columns=["model", "cycle", "epoch", "losses"])
    train_output_file = Path(output_path, "training_progress.tsv")
    tmp_train_df.to_csv(train_output_file, mode='w', sep='\t', header=True)
    del tmp_train_df

    tmp_test_df = pd.DataFrame(columns=["model", "cycle", "epoch", "loss", "accuracy"])
    test_output_file = Path(output_path, "testing_progress.tsv")
    tmp_test_df.to_csv(test_output_file, mode='w', sep='\t', header=True)
    del tmp_test_df

    # Prepare our data for the simple CNN
    training_data = CIFAR100Coarse(
        root="data",
        train=True,
        download=True,
        transform=data_managers.train_transform
    )

    testing_data = CIFAR100Coarse(
        root="data",
        train=False,
        download=True,
        transform=data_managers.test_transform
    )

    # Lock the random seed here to make the batches are consistent between tests with different models
    torch.manual_seed(36246)
    # Split the training data into sets of 5000 entries
    training_sets = random_split(training_data, [5000] * 10)
    # Split the testing data into sets of 1000 entries
    testing_sets = random_split(testing_data, [1000] * 10)
    # Restore the random seed back to normal for the training
    torch.initial_seed()

    # Run the simple CNN 5 times
    for cycle in range(10):
        base_data = {
            "model": ["SimpleCNN"],
            "cycle": [cycle]
        }

        train_dataloader = DataLoader(training_sets[cycle], batch_size=batch_size, shuffle=True)
        testing_dataloader = DataLoader(testing_sets[cycle], batch_size=batch_size, shuffle=True)

        # Initialize our model
        model = SimpleCNN(3, 20).to(device)

        # Initialize our loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.99, 0.999), weight_decay=0.001)

        # Run the train/test cycles
        for e in range(epochs):
            base_data['epoch'] = [e]

            print(f"Epoch {e + 1} (Simple CNN)\n----------------------------------------------------")
            losses = model.run_train_cycle(train_dataloader, loss_fn, optim)
            training_df = pd.DataFrame({**base_data, "losses": [losses]})
            training_df.to_csv(train_output_file, mode='a', header=False, sep='\t')
            del training_df

            test_results = model.run_test_cycle(testing_dataloader, loss_fn)
            testing_df = pd.DataFrame({**base_data, **test_results})
            testing_df.to_csv(test_output_file, mode='a', header=False, sep='\t')
            del testing_df
        print(f"Finished cycle {cycle}!")

    # Prepare our data for the paired CNN/Dense network
    training_data = data_managers.build_dataset(True)
    testing_data = data_managers.build_dataset(False)

    # Split the training data into sets of 5000 entries
    training_sets = random_split(training_data, [5000] * 10)
    # Split the testing data into sets of 1000 entries
    testing_sets = random_split(testing_data, [1000] * 10)

    # Run the paired CNN+Dense paired network 5 times
    for cycle in range(10):
        base_data = {
            "model": "PairedDenseCNN",
            "cycle": cycle
        }

        train_dataloader = DataLoader(training_sets[cycle], batch_size=batch_size)
        testing_dataloader = DataLoader(testing_sets[cycle], batch_size=batch_size)

        # Initialize our model
        model = PairedDenseCNN(3, 3, 20).to(device)

        # Initialize our loss function and optimizer
        loss_fn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.99, 0.999))

        # Run the train/test cycles
        for e in range(epochs):
            base_data['epoch'] = e

            print(f"Epoch {e + 1} (Paired CNN/Dense)\n----------------------------------------------------")
            losses = model.run_train_cycle(train_dataloader, loss_fn, optim)
            training_df = pd.DataFrame({**base_data, "losses": [losses]})
            training_df.to_csv(train_output_file, mode='a', header=False, sep='\t')
            del training_df

            test_results = model.run_test_cycle(testing_dataloader, loss_fn)
            testing_df = pd.DataFrame({**base_data, **test_results})
            testing_df.to_csv(test_output_file, mode='a', header=False, sep='\t')
            del testing_df
        print(f"Finished cycle {cycle}!")
