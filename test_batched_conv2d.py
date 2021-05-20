import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import data_managers
from shared_utils import run_train_cycle, run_test_cycle, initialize_output_files, write_new_result
from test_conv2d import Conv2dNet

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

    # Split the training data into sets of 5000 entries
    training_sets = random_split(training_data, [5000] * 10)
    # Split the testing data into sets of 1000 entries
    testing_sets = random_split(testing_data, [1000] * 10)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    for c in range(no_cycles):
        # Rebuild the model and its optimizer
        # Build the model, initialized with one column
        model = Conv2dNet().to(device)
        optim = torch.optim.Adam(model.parameters(), lr=train_lr,
                                 betas=(0.99, 0.999), weight_decay=0.001)
        # Train the model, tracking the results
        train_dataloader = DataLoader(training_sets[c], batch_size=batch_size, shuffle=True)
        testing_dataloader = DataLoader(testing_sets[c], batch_size=batch_size, shuffle=True)
        for e in range(train_epochs):
            print(f"=============Cycle {c} Epoch {e}=============")
            train_result = run_train_cycle(model, train_dataloader, loss_fn, optim, device)
            write_new_result(c, e, train_result['accuracy'],
                             train_result['loss'], out_path, is_train=True)

            test_result = run_test_cycle(model, testing_dataloader, loss_fn, device)
            write_new_result(c, e, test_result['accuracy'],
                             test_result['loss'], out_path, is_train=False)
