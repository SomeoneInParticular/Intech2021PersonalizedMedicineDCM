import argparse
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split

import data_managers
from cifar100coarse.cifar100coarse import CIFAR100Coarse
from convnet_progressive import ProgressiveNN
from dense import build_init_blocks, build_block_gens, gen_classif_block, run_train_cycle, run_test_cycle


if __name__ == '__main__':
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

    # The layer counts for each block (based on DenseNet-169)
    block_config = (6, 12, 36, 36)

    # The set of dense block parameters (for initial blocks)
    init_blocks = build_init_blocks(block_config)

    # Initialize the block generator with a convolve of half that size
    block_gens = build_block_gens(block_config)

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

    for cycle in range(10):
        print("==========================================================================")
        print(f"Beginning Cycle {cycle}")
        print("==========================================================================")

        # Rebuild the model
        model = ProgressiveNN(input_shape=(3, 32, 32), initial_blocks=init_blocks,
                              block_generators=block_gens, output_builder=gen_classif_block,
                              update_trained=True).to(device)

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
        optim = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001, momentum=0.9, dampening=0)

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
