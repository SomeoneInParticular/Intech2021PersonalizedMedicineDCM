import argparse
from pathlib import Path

import copy
import data_managers
import numpy as np
import pandas as pd
import torch

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from convnet_progressive import ProgressiveNN

from cifar100coarse.cifar100coarse import CIFAR100Coarse


def run_progressive_train_cycle(model, dataloader, loss_fn, optim, device, report_rate=50):
    """
    Run a training cycle on a model
    :param model: The model to run the training cycle on
    :param dataloader: The dataloader which should be used to pull train/validate data from
    :param loss_fn: The loss function to use during evaluation
    :param optim: The optimizer to do the actual model improvement
    :param device: The device the training should be done on
    :param report_rate: How often the current loss of the model should be reported (via console output)
    :return: A list of losses over the training period, for diagnostic/plotting purposes
    """
    losses = []

    for i, (X, y) in enumerate(dataloader):
        # Load the data into the device for processing
        X, y = X.to(device), y.to(device)

        # Prediction loss calc
        pred = model(data=X)
        loss = loss_fn(pred, y)
        losses.append(float(loss.cpu().detach().numpy()))

        # Backpropagation to update the model
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Report loss every 100 batches
        if i % report_rate == 0:
            print(f'Loss: {loss.item()} for batch {i}')

    return losses


def run_progressive_test_cycle(model, dataloader, loss_fn, device):
    """
    Run a training cycle on a model
    :param model: The model to run the training cycle on
    :param dataloader: The dataloader which should be used to pull train/validate data from
    :param loss_fn: The loss function to use during evaluation
    :param device: The device the testing should be done on
    :return: A list of losses over the training period, for diagnostic/plotting purposes
    """
    # Set this to evaluate mode
    model.eval()
    # Accumulate loss and accuracy across the dataset
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
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


if __name__ == '__main__':
    from convnet import conv_chain, pooled_conv_chain

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description="Run our static networks and record their progress")

    parser.add_argument('-o', '--output', help='The destination in which to place the output files')
    parser.add_argument('-et', '--epochs_train', help='The number of epochs that should be run for each training cycle',
                        type=int)
    parser.add_argument('-ep', '--epochs_prune', help='The number of epochs that should be run for each pruning cycle',
                        type=int)
    parser.add_argument('-mp', '--max_prune', help='The maximum number of pruning cycles that should be run before '
                                                   'proceeding to the next progression stage',
                        type=int)
    parser.add_argument('-b', '--batch', help='The size of the batches to submit to the network during training',
                        type=int)

    args = parser.parse_args()

    output_path = args.output

    train_epochs = args.epochs_train
    prune_epochs = args.epochs_prune
    max_prune_cycles = args.max_prune
    batch_size = args.batch

    del args
    del parser

    # Build the components of the progressive learning system
    init_blocks = [
        nn.Sequential(
            *conv_chain(3, 32)
        ),
        nn.Sequential(
            *pooled_conv_chain(32, 32, 0.25)
        ),
        nn.Sequential(
            *conv_chain(32, 64)
        ),
        nn.Sequential(
            *pooled_conv_chain(64, 64, 0.25)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(1600, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    ]

    block_gens = [
        lambda in_size: nn.Sequential(
            *conv_chain(in_size[0], 16)
        ),
        lambda in_size: nn.Sequential(
            *pooled_conv_chain(in_size[0], 16, 0.25)
        ),
        lambda in_size: nn.Sequential(
            *conv_chain(in_size[0], 16)
        ),
        lambda in_size: nn.Sequential(
            *pooled_conv_chain(in_size[0], 32, 0.25)
        ),
        lambda in_size: nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(in_size), 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    ]

    def softmax_out_generator(in_shape: np.array):
        return nn.Sequential(
            nn.Linear(in_shape[0], 20),
            nn.Softmax(dim=1)
        )

    # Identify whether a CUDA GPU is available to run the analyses on
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare our data for the model
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

    # Lock the random seed here to make the batches consistent between tests
    torch.manual_seed(36246)
    # Split the training data into sets of 5000 entries
    training_sets = random_split(training_data, [5000]*10)
    # Split the testing data into sets of 1000 entries
    testing_sets = random_split(testing_data, [1000]*10)
    # Restore the random seed back to normal for the training
    torch.initial_seed()

    # Initialize our dataframe dump files, resetting them if they already exist
    tmp_train_df = pd.DataFrame(columns=["model", "cycle", "epoch", "losses"])
    train_output_file = Path(output_path, "training_progress.tsv")
    tmp_train_df.to_csv(train_output_file, mode='w', sep='\t', header=True)
    del tmp_train_df

    tmp_test_df = pd.DataFrame(columns=["model", "cycle", "epoch", "loss", "accuracy"])
    test_output_file = Path(output_path, "testing_progress.tsv")
    tmp_test_df.to_csv(test_output_file, mode='w', sep='\t', header=True)
    del tmp_test_df

    # Initialize our model
    model = ProgressiveNN((3, 32, 32), initial_blocks=init_blocks, block_generators=block_gens,
                          output_builder=softmax_out_generator, update_trained=True).to(device)

    for cycle in range(10):
        print("==========================================================================")
        print(f"Beginning Cycle {cycle}")
        print("==========================================================================")

        base_data = {
            "model": ["Progressive"],
            "cycle": [cycle]
        }

        train_dataloader = DataLoader(training_sets[cycle], batch_size=batch_size, shuffle=True)
        testing_dataloader = DataLoader(testing_sets[cycle], batch_size=batch_size, shuffle=True)

        # Initialize the loss and optimization functions
        loss_fn = nn.CrossEntropyLoss()
        train_optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.99, 0.999), weight_decay=0.001)

        print("Beginning Training")

        def run_epochs(target_model, epoch_no, optim, loss):
            # Runs epoch_no epochs on the target model, using the provided optimizer to update the model and
            # the provided loss function to evaluate how "good" it during evaluation
            results = []
            for e in range(epoch_no):
                print(f"Epoch {e + 1} (Progressive)\n----------------------------------------------------")

                base_data['epoch'] = [e]
                losses = run_progressive_train_cycle(target_model, train_dataloader, loss, optim, device)
                training_df = pd.DataFrame({**base_data, "losses": [losses]})
                training_df.to_csv(train_output_file, mode='a', header=False, sep='\t')

                test_results = run_progressive_test_cycle(target_model, testing_dataloader, loss, device)
                testing_df = pd.DataFrame({**base_data, **test_results})
                testing_df.to_csv(test_output_file, mode='a', header=False, sep='\t')
                results.append(test_results)
            return results

        # Run the training epochs for the set
        results = run_epochs(model, train_epochs, train_optim, loss_fn)

        # Prune the trained network
        print(f"Beginning pruning and fine-tuning")
        prior_acc = 0
        delta_acc = 0
        prune_iter = 0
        while delta_acc >= 0 and prune_iter < max_prune_cycles:
            print(f"Prune Iteration: {prune_iter}")
            # noinspection PyRedeclaration
            prior_state = copy.deepcopy(model.state_dict())

            model.prune()

            tune_optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.99, 0.999), weight_decay=0.001)

            result = run_epochs(model, prune_epochs, tune_optim, loss_fn)

            current_acc = np.max([r['accuracy'][0] for r in result])
            delta_acc = current_acc - prior_acc
            prior_acc = current_acc
            prune_iter += 1

        # Restore the finale state dict
        # noinspection PyUnboundLocalVariable
        if len(prior_state) != 0 and delta_acc < 0:
            # noinspection PyTypeChecker
            model.load_state_dict(prior_state)

        # Progress the network
        model.progress(device=device)
