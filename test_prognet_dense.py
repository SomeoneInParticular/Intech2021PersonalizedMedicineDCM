import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models.densenet import _DenseBlock, _Transition

import data_managers
from prognet import ProgBlock, ProgColumn, ProgNet
from shared_utils import run_train_cycle, run_test_cycle, initialize_output_files, default_lateral, write_new_result, \
    default_l1_prune


class ProgDenseInitBlock(ProgBlock):
    def __init__(self, in_shape, no_init_features):
        module = nn.Sequential(
            nn.Conv2d(in_shape[0], no_init_features, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(no_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        super().__init__(module, default_lateral)


class ProgDenseBlock(ProgBlock):
    def __init__(self, in_shape, no_layers, bn_size, growth_rate, drop_rate):
        module = _DenseBlock(
            num_layers=no_layers, num_input_features=in_shape[0],
            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        super().__init__(module, default_lateral)


class ProgDenseWithTransitionBlock(ProgBlock):
    def __init__(self, in_shape, no_layers, bn_size, growth_rate, drop_rate):
        dense_block = _DenseBlock(
            num_layers=no_layers, num_input_features=in_shape[0],
            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        new_input_count = in_shape[0] + no_layers * growth_rate
        transition_block = _Transition(
            num_input_features=new_input_count,
            num_output_features=new_input_count//2
        )
        module = nn.Sequential(
            dense_block,
            transition_block
        )
        super().__init__(module, default_lateral)


class ProgBatchNormBlock(ProgBlock):
    def __init__(self, in_shape):
        print(in_shape[0])
        module = nn.BatchNorm2d(in_shape[0])
        super().__init__(module, default_lateral)


class ProgDenseClassifier(ProgBlock):
    def __init__(self, in_shape, no_classes):
        module = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1),
            nn.Linear(np.prod(in_shape), no_classes),
            # nn.Softmax()
        )
        super().__init__(module, default_lateral)


class ProgDenseNetColumn(ProgColumn):
    def __init__(self, parents, in_shape, growth_rate,
                 no_init_features, block_config, no_classes):
        # The list of blocks to contain within this column
        block_list = []
        # A row index tracker to keep tabs on the currently active row
        row_index = 0

        # Run everything that follows without a gradient to save some work
        with torch.no_grad():
            # This column opens with an initial convolve block
            init_block = ProgDenseInitBlock(in_shape, no_init_features)
            block_list.append(init_block)

            # Initialize a "dummy" dataset, which will help build future blocks
            if len(parents) > 0:
                no_elems = parents[0].last_output_list[row_index].shape[0]
            else:
                no_elems = 64
            x = init_block(torch.rand(no_elems, *in_shape))

            # Subsequent blocks are built using the block config
            for i, no_layers in enumerate(block_config):
                x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
                in_shape = x.shape[1:]
                if i >= len(block_config)-1:
                    new_block = ProgDenseBlock(
                        in_shape, no_layers, 4, growth_rate, 0.2
                    )
                else:
                    new_block = ProgDenseWithTransitionBlock(
                        in_shape, no_layers, 4, growth_rate, 0.2
                    )
                block_list.append(new_block)
                x = new_block(x)
                row_index += 1

            # The final batch norm block
            x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
            in_shape = x.shape[1:]
            batchnorm_block = ProgBatchNormBlock(in_shape)
            block_list.append(batchnorm_block)
            x = batchnorm_block(x)
            row_index += 1

            # The classifier block
            x = torch.cat([x, *self.build_prior_tensor(parents, row_index)], dim=1)
            in_shape = x.shape[1:]
            classif_block = ProgDenseClassifier(in_shape, no_classes)
            block_list.append(classif_block)

        super().__init__(block_list, parents)


    @staticmethod
    def build_prior_tensor(parents, index: int):
        if parents is not None and len(parents) > 0:
            return [p.last_output_list[index].to("cpu") for p in parents]
        else:
            return []


class ProgDenseNet(ProgNet):
    def gen_new_col(self, parents):
        return ProgDenseNetColumn(
            parents=parents, in_shape=self.in_shape, growth_rate=16,
            block_config=(6, 12, 32, 32), no_init_features=32,
            no_classes=self.no_classes
        )

    def __init__(self, in_shape, device='cpu', no_classes=10):
        self.in_shape = in_shape
        self.no_classes = no_classes
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
    model = ProgDenseNet(in_shape=in_shape, device=device, no_classes=10)
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
            default_l1_prune(model)
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
