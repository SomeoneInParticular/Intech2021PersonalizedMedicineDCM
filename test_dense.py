import argparse
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import densenet169

import data_managers
from shared_utils import run_train_cycle, run_test_cycle, initialize_output_files, write_new_result


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
    train_epochs = 300
    prune_epochs = 10
    in_shape = (3, 32, 32)
    batch_size = 64

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

    # Build the optimizer and loss function for the training
    loss_fn = nn.CrossEntropyLoss()

    for c in range(no_cycles):
        # Build the model, initialized with one column
        model = densenet169().to(device)

        # Rebuild the optimizer and the scheduler to reset their values
        optim = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9, dampening=0)
        scheduler = MultiStepLR(optim, milestones=[
            train_epochs // 2,
            train_epochs * 3 // 4
        ], gamma=0.1)

        # Train the model, tracking the results
        for e in range(train_epochs):
            print(f"=============Cycle {c} Epoch {e}=============")
            train_result = run_train_cycle(model, train_dataloader,
                                           loss_fn, optim, device)
            write_new_result(c, e, train_result['accuracy'],
                             train_result['loss'], out_path, is_train=True)

            test_result = run_test_cycle(model, testing_dataloader,
                                         loss_fn, device)
            write_new_result(c, e, test_result['accuracy'],
                             test_result['loss'], out_path, is_train=False)

            scheduler.step()
