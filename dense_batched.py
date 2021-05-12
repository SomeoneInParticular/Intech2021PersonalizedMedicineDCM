import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from torchvision.models.densenet import _DenseBlock, _Transition

import data_managers
from cifar100coarse.cifar100coarse import CIFAR100Coarse
from convnet_progressive import ProgressiveNN


def init_convolve(no_features):
    return nn.Sequential(OrderedDict([
        ('conv0', nn.Conv2d(3, no_features, kernel_size=(7, 7), stride=(2, 2),
                            padding=(3, 3), bias=False)),
        ('norm0', nn.BatchNorm2d(no_features)),
        ('relu0', nn.ReLU(inplace=True)),
        ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
    ]))


def run_train_cycle(model, dataloader, loss_fn, optim, device, report_rate=50):
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


def run_test_cycle(model, dataloader, loss_fn, device):
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
    # Parse the command line arguments
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

    # Identify whether a CUDA GPU is available to run the analyses on
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the initial block list with 64 features
    init_blocks = [
        init_convolve(64)
    ]

    # The layer counts for each block (based on DenseNet-169)
    block_config = (6, 12, 36, 36)
    # The set of dense block parameters (for initial blocks)
    no_features = 64
    growth_rate = 32
    # Build up the set of initial blocks for the model
    for i, no_layers in enumerate(block_config):
        dense_block = _DenseBlock(
            num_layers=no_layers,
            num_input_features=no_features,
            bn_size=4,
            growth_rate=growth_rate,
            drop_rate=0
        )
        no_features = no_features + no_layers * growth_rate
        if i < len(block_config)-1:
            trans_block = _Transition(num_input_features=no_features,
                                      num_output_features=no_features // 2)
            full_block = nn.Sequential(dense_block, trans_block)
            init_blocks.append(full_block)
            no_features = no_features // 2
        else:
            init_blocks.append(dense_block)

    # The final batch norm
    init_blocks.append(nn.BatchNorm2d(no_features))

    # A function generate Dense block generator functions
    def dense_block_func_gen(no_layers: int, growth_rate: int,
                             bn_size: int, drop_rate: float,
                             should_transition: bool = True):
        def dense_block_gen(in_shape):
            no_features = in_shape[0]
            dense_block = _DenseBlock(
                num_layers=no_layers,
                num_input_features=no_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            if should_transition:
                # Determine the output shape directly
                x = torch.rand(256, *in_shape)
                no_features = dense_block(x).shape[1]  # [0] = 256, being our sample count
                trans_block = _Transition(num_input_features=no_features,
                                          num_output_features=no_features // 2)
                full_block = nn.Sequential(dense_block, trans_block)
                return full_block
            else:
                return dense_block

        return dense_block_gen

    # Halved growth rate for subsequent blocks added via progression
    growth_rate = 16

    # Initialize the block generator with a convolve of half that size
    block_gens = [
        lambda in_shape: init_convolve(32)
    ]

    # Add the remaining block generators for this model
    for i, no_layers in enumerate(block_config):
        should_transition = i < len(block_config)-1
        new_gen = dense_block_func_gen(no_layers, growth_rate,
                                       4, 0, should_transition)
        block_gens.append(new_gen)

    block_gens.append(
        lambda in_shape: nn.BatchNorm2d(in_shape[0])
    )

    # The output block generator
    no_classes = 20

    def gen_classif_block(in_shape: np.array):
        return nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(np.prod(in_shape), no_classes),
            nn.Softmax(dim=1)  # To match the setup of the ConvNet
        )

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
    training_sets = random_split(training_data, [5000] * 10)
    # Split the testing data into sets of 1000 entries
    testing_sets = random_split(testing_data, [1000] * 10)
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

    # Finally, build the model
    model = ProgressiveNN(input_shape=(3, 32, 32), initial_blocks=init_blocks,
                          block_generators=block_gens, output_builder=gen_classif_block,
                          update_trained=True).to(device)

    for cycle in range(2):
        print("==========================================================================")
        print(f"Beginning Cycle {cycle}")
        print("==========================================================================")

        base_data = {
            "model": ["Progressive"],
            "cycle": [cycle]
        }

        train_dataloader = DataLoader(training_sets[cycle], batch_size=batch_size, shuffle=True)
        testing_dataloader = DataLoader(testing_sets[cycle], batch_size=batch_size, shuffle=True)

        # Initialize the loss function
        loss_fn = nn.CrossEntropyLoss()

        print(f"Beginning Training")
        # Initialize the optimizer and its attached scheduler
        optim = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.99, 0.999), weight_decay=0.001)

        scheduler = MultiStepLR(optim, milestones=[
            train_epochs // 2,
            train_epochs * 3 // 4
        ], gamma=0.1)

        def run_epochs(target_model, epoch_no, optim, scheduler, loss):
            # Runs epoch_no epochs on the target model, using the provided optimizer to update the model and
            # the provided loss function to evaluate how "good" it during evaluation
            results = []
            for e in range(epoch_no):
                print(f"Epoch {e + 1}\n----------------------------------------------------")

                # Training
                base_data['epoch'] = [e]
                losses = run_train_cycle(target_model, train_dataloader, loss, optim, device)
                training_df = pd.DataFrame({**base_data, "losses": [losses]})
                training_df.to_csv(train_output_file, mode='a', header=False, sep='\t')

                # Testing/Validation
                test_results = run_test_cycle(target_model, testing_dataloader, loss, device)
                testing_df = pd.DataFrame({**base_data, **test_results})
                testing_df.to_csv(test_output_file, mode='a', header=False, sep='\t')
                results.append(test_results)

                # Scheduler update
                scheduler.step()
            return results

        # Run the training epochs for the set
        results = run_epochs(model, train_epochs, optim, scheduler, loss_fn)
