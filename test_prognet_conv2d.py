import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import data_managers
from prognet import ProgBlock, ProgColumn, ProgNet
from shared_utils import default_lateral, default_l1_prune, \
    initialize_output_files, run_train_cycle, run_test_cycle, write_new_result


class ProgConv2dBlock(ProgBlock):
    def __init__(self, in_shape, out_shape):
        module = nn.Sequential(
            nn.Conv2d(in_shape[0], out_shape, (3, 3), (1, 1)),
            nn.BatchNorm2d(out_shape),
            nn.ReLU(inplace=True)
        )
        super().__init__(module, default_lateral)


class ProgPoolingConv2dBlock(ProgBlock):
    def __init__(self, in_shape, out_shape, drop_rate):
        module = nn.Sequential(
            nn.Conv2d(in_shape[0], out_shape, (3, 3), (1, 1)),
            nn.BatchNorm2d(out_shape),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 2)),
            nn.Dropout(drop_rate)
        )
        super().__init__(module, default_lateral)


class ProgLinearBlock(ProgBlock):
    def __init__(self, in_shape, out_shape, drop_rate):
        module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(in_shape), out_shape),
            nn.BatchNorm1d(out_shape),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate)
        )
        super().__init__(module, default_lateral)


class ProgOutputClassifBlock(ProgBlock):
    def __init__(self, in_shape, no_classes):
        module = nn.Sequential(
            nn.Linear(in_shape[0], no_classes),
            nn.Softmax(dim=1)
        )
        super().__init__(module, default_lateral)


class ProgConvNetColumn(ProgColumn):
    def __init__(self, parents, in_shape, conv_drop=0.25, lin_drop=0.5,
                 out_shape1=16, out_shape2=16, out_shape3=16,
                 out_shape4=32, out_shape5=256, no_classes=20):
        # The list of blocks to contain within this column
        block_list = []
        # Initialize a row index to keep track of where we are in the parent list
        row_index = 0

        # Run everything that follows without a gradient to save some memory
        with torch.no_grad():
            # First block is straightforward, just initializing a Conv2dBlock
            # which accepts the input size
            b1 = ProgConv2dBlock(in_shape, out_shape1)
            block_list.append(b1)

            # Initialize a "dummy" dataset, which will help build future blocks
            if len(parents) > 0:
                no_elems = parents[0].last_output_list[row_index].shape[0]
            else:
                no_elems = 64
            x = b1(torch.rand(no_elems, *in_shape))

            # Second blocks accepts concatenated output from all prior blocks into a PooledConvBlock
            x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
            in_shape = x.shape[1:]
            b2 = ProgPoolingConv2dBlock(in_shape, out_shape2, conv_drop)
            block_list.append(b2)
            row_index += 1

            # Third block is the same, but uses a basic convnet block
            x = b2(x)
            x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
            in_shape = x.shape[1:]
            b3 = ProgConv2dBlock(in_shape, out_shape3)
            block_list.append(b3)
            row_index += 1

            # Fourth block is another pooled convnet block
            x = b3(x)
            x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
            in_shape = x.shape[1:]
            b4 = ProgPoolingConv2dBlock(in_shape, out_shape4, conv_drop)
            block_list.append(b4)
            row_index += 1

            # Fifth block is a linear interpretation block
            x = b4(x)
            x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
            in_shape = x.shape[1:]
            b5 = ProgLinearBlock(in_shape, out_shape5, lin_drop)
            block_list.append(b5)
            row_index += 1

            # Output block is a softmax classifier
            x = b5(x)
            x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
            in_shape = x.shape[1:]
            b6 = ProgOutputClassifBlock(in_shape, no_classes)
            block_list.append(b6)

        super().__init__(block_list, parents)

    @staticmethod
    def build_prior_tensor(parents, index: int):
        if parents is not None and len(parents) > 0:
            return [p.last_output_list[index].to("cpu") for p in parents]
        else:
            return []


class ProgConv2dNet(ProgNet):
    def gen_new_col(self, parents):
        return ProgConvNetColumn(
            parents, self.in_shape, conv_drop=0.25, lin_drop=0.5,
            out_shape1=16, out_shape2=16, out_shape3=16,
            out_shape4=32, out_shape5=256, no_classes=20
        )

    def __init__(self, in_shape, device="cpu"):
        self.in_shape = in_shape
        super().__init__(column_generator=self.gen_new_col, device=device)

    def progress(self):
        self.freeze_all_columns()
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
    train_epochs = 90
    prune_epochs = 10
    in_shape = (3, 32, 32)
    train_lr = 0.001
    prune_lr = 0.0001
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

    # Build the model, initialized with one column
    model = ProgConv2dNet(device=device, in_shape=in_shape)
    model.add_column(ProgConvNetColumn(
        model.columns, in_shape, conv_drop=0.25, lin_drop=0.5,
        out_shape1=32, out_shape2=32, out_shape3=64,
        out_shape4=64, out_shape5=512, no_classes=20
    ))

    # Build the optimizer and loss function for the training
    loss_fn = nn.CrossEntropyLoss()

    for c in range(no_cycles):
        # Rebuild the optimizer with the new model's parameters
        optim = torch.optim.Adam(model.parameters(), lr=train_lr,
                                 betas=(0.99, 0.999), weight_decay=0.001)

        # Train the model, tracking the results
        for e in range(train_epochs):
            print(f"=============Cycle {c} Epoch {e}=============")
            train_result = run_train_cycle(model, train_dataloader, loss_fn, optim, device)
            write_new_result(c, e, train_result['accuracy'],
                             train_result['loss'], out_path, is_train=True)

            test_result = run_test_cycle(model, testing_dataloader, loss_fn, device)
            write_new_result(c, e, test_result['accuracy'],
                             test_result['loss'], out_path, is_train=False)

        # Save the final accuracy explicitly
        prior_acc = 0

        # Rebuild the optimizer with 1/10th the learning rate
        optim = torch.optim.Adam(model.parameters(), lr=prune_lr,
                                 betas=(0.99, 0.999), weight_decay=0.001)

        # Prune the model iteratively
        prior_gain = 0
        p = 0
        prior_state = None
        while prior_gain >= 0:
            prior_state = copy.deepcopy(model.state_dict())
            default_l1_prune(model)
            results = []
            for e in range(prune_epochs):
                print(f"=============Cycle {c} Prune {p} Epoch {e}=============")
                train_result = run_train_cycle(model, train_dataloader, loss_fn, optim, device)
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

        # Progress the model
        model.progress()
