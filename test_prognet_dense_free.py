import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import data_managers
from test_prognet_dense import ProgDenseNet, ProgDenseNetColumn
from shared_utils import run_train_cycle, run_test_cycle, initialize_output_files, write_new_result, \
    default_full_l1_prune


class ProgDenseFreeNet(ProgDenseNet):
    def progress(self):
        # self.freeze_all_columns()
        self.add_column()


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
    train_lr = 0.001
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

    # Build the model, initialized with one column
    model = ProgDenseFreeNet(in_shape=in_shape, device=device, no_classes=10)
    model.add_column(ProgDenseNetColumn(
        model.columns, in_shape, 32, 64, (6, 12, 32, 32), 10
    ))

    for c in range(no_cycles):
        # Build the optimizer and loss function for the training
        train_optim = torch.optim.SGD(model.parameters(), lr=0.1,
                                      weight_decay=0.0001, momentum=0.9,
                                      dampening=0)
        scheduler = MultiStepLR(train_optim, milestones=[
            train_epochs // 2,
            train_epochs * 3 // 4
        ], gamma=0.1)
        loss_fn = nn.CrossEntropyLoss()

        # Train the model, tracking the results
        for e in range(train_epochs):
            print(f"=============Cycle {c} Epoch {e}=============")
            train_result = run_train_cycle(model, train_dataloader, loss_fn, train_optim, device)
            write_new_result(c, e, train_result['accuracy'],
                             train_result['loss'], out_path, is_train=True)

            test_result = run_test_cycle(model, testing_dataloader, loss_fn, device)
            write_new_result(c, e, test_result['accuracy'],
                             test_result['loss'], out_path, is_train=False)

            scheduler.step()

        # Save the final accuracy explicitly
        prior_acc = 0

        # Rebuild the optimizer with 1/10th the lowest learning rate
        prune_optim = torch.optim.SGD(model.parameters(), lr=0.0001,
                                      weight_decay=0.0001, momentum=0.9,
                                      dampening=0)

        # Prune the model iteratively
        prior_gain = 0
        p = 0
        prior_state = copy.deepcopy(model.state_dict())
        while prior_gain >= 0:
            prior_state = copy.deepcopy(model.state_dict())
            default_full_l1_prune(model)
            results = []
            for e in range(prune_epochs):
                print(f"=============Cycle {c} Prune {p} Epoch {e}=============")
                train_result = run_train_cycle(model, train_dataloader, loss_fn, prune_optim, device)
                write_new_result(c, e, train_result['accuracy'],
                                 train_result['loss'], out_path, is_train=True)

                test_result = run_test_cycle(model, testing_dataloader, loss_fn, device)
                write_new_result(c, e, test_result['accuracy'],
                                 test_result['loss'], out_path, is_train=False)
                results.append(test_result)
            new_acc = np.mean([r['accuracy'] for r in results])
            prior_gain = new_acc - prior_acc
            prior_acc = new_acc
            p += 1

        # Restore the state to the last (best) option, if successful at all
        if prior_state is not None and len(prior_state) > 0 and prior_gain < 0:
            # noinspection PyTypeChecker
            model.load_state_dict(prior_state)

        model.progress()
