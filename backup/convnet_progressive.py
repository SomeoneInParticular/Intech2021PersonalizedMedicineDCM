import argparse
from pathlib import Path

import copy
import data_managers
import numpy as np
import pandas as pd
import torch

from itertools import chain
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import prune
from typing import Callable, Tuple, Union, Iterable

from cifar100coarse.cifar100coarse import CIFAR100Coarse


class L1UnstructuredMin1(prune.L1Unstructured):
    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int, else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        # If amount is not 0, but less than 1, round up to 1
        if 0 < nparams_toprune < 1:
            nparams_toprune = 1
        # This should raise an error if the number of units to prune is larger than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask


class ProgressiveNN(nn.Module):
    # Type hint for pruning method generating callable
    PruningMethodGenerator = Callable[[Iterable[Tuple[nn.Module, str]], Union[float, int]], prune.BasePruningMethod]

    # Class constants
    OUT_LAYER_NAME = "output"

    def __init__(self, input_shape: Tuple[int, ...],
                 block_generators: list[Callable[[np.array], nn.Module]],
                 initial_blocks: list[nn.Module] = None,
                 output_builder: Callable[[np.array], nn.Module] = None,
                 pruning_method: PruningMethodGenerator = None,
                 update_trained: bool = False):
        """
        Initialize a model which follows the Progressive Learning framework (Fayek et. al, 2020).
        Also used as to track and manage branches for the BranchedProgressiveNN (WIP).
        :param input_shape: The shape of a *single* element expected to be supplied to this model
        :param block_generators: A list of callables which will generate new blocks during the progression stage of
            training. Each callable should accept one argument, being a numpy array of integers of any size, which
            represents the output shape of a *single* tensor from the prior layer (which will grow over time), and
            return a module which can accept tensors of that shape. Note that the first module produced by this set of
            callables will be the one which receives input data!
        :param initial_blocks: Optional list of torch modules to start the neural network with. Currently this is
            required to be the same size as the block generators list prior, though this will probably change in the
            future. If not provided, the block generator will generate the first set of blocks.
        :param output_builder: A callable which, given the shape of the feeding layers concatenated outputs (in the form
            of an numpy array of integers), returns a module that will produce the predictions of the overall model.
            Note that this will be regenerated from scratch every time progression occurs; as a result, we recommend
            that the results of this function should not rely on trainable parameters (this may change in the future).
            If none is provided, the forward run will simply return a concatenated list of all output tensors from all
            blocksets.
        :param pruning_method: A callable which accepts an iterable of (module, name) tuples (being the active modules
            of the network) and either a float (proportion) or int (hard amount), and returns a pruning method for the
            pruning stage of the training. If not provided it will default to a method of pruning 10% of the lowest
            magnitude weights (L1 norm) across the entire model's weights.
        :param update_trained: Whether previously trained blocks should be allowed to update. By default this should
            remain disabled to avoid forgetting trends learned in previous cycles, but can be enabled if one does not
            need backwards recall and is willing to accept the higher risk of over-fitting.
        """
        super().__init__()
        # Track whether prior blocks (which have already been trained) should be updated during further training
        self.update_trained = update_trained
        # Track the number of input features the model expects
        self.input_shape = input_shape
        # The list of block generators for the progression step
        self._block_generators = block_generators

        # Whether this branch should be frozen (and no longer progress).
        self._frozen = False

        # The "output" layer, which is regenerated every progression step
        self._output_builder = None
        if output_builder is not None:
            self._output_builder = output_builder

        # A list of block lists, one list per prior progression step
        self._prior_blocks = []

        # Initialize the initial set of blocks if they were not provided
        if initial_blocks is None:
            self.progress()
            # A list of the output sizes each layer. Required for efficient progression
            self._layer_output_shapes = [(0,) for i in range(len(block_generators))]
        # Otherwise, confirm the block generator set is the same length as the initial blocks (TEMPORARY)
        else:
            assert len(initial_blocks) == len(block_generators), \
                f"The length of the initial blocks ({len(initial_blocks)}) should " \
                f"equal the length of the block generators ({len(block_generators)})"
            # Save the list of current blocks for layer management, alongside their output size
            self._current_blocks = initial_blocks
            for i, block in enumerate(self._current_blocks):
                self.add_module(f"blockset_0_layer_{i}", block)
            # Store their output shapes for use during progression

            # Generate the first set of per-block output shapes
            self._layer_output_shapes = []
            x = torch.rand(256, *input_shape)
            for b in initial_blocks:
                x = b(x)
                self._layer_output_shapes.append(x.shape[1:])
            # Generate the final output block, if a generator was provided
            # (otherwise it will just concat everything along axis 1)
            self.output_layer = None
            if output_builder is not None:
                self.output_layer = output_builder(x.shape[1:])
                self.add_module(ProgressiveNN.OUT_LAYER_NAME, self.output_layer)

        # The pruning methodology to employ on the branch
        if pruning_method is None:
            pruning_method = ProgressiveNN.default_l1_pruning
        self._pruning_method = pruning_method

        # Track whether the network is currently in pruning mode (being in progression mode otherwise)
        self.is_pruning = False

    def progress(self, device: str = "cpu", warn: bool = False):
        """
        Generates a new set of blocks for the model, updating variables to match
        :param device: The device to load the new modules onto
        :param warn: Whether to warn the user if the branch fails to progress
        """
        # If the branch has been frozen, return early and warn the user
        if self._frozen:
            if warn:
                print(f"WARNING: Branch '{self._get_name()}' is frozen, and was not progressed.")
            return

        # If we were in the pruning stage, finalize the pruned parameters and switch the stage back to progression
        if self.is_pruning:
            # Disable training for all if we are not allowed to update prior knowledge
            if not self.update_trained:
                for _, param in self.named_parameters():
                    param.requires_grad = False
            # Mark the mode as no longer pruning
            self.is_pruning = False

        # Store the last set of actively trainable blocks into the prior cache
        if self._current_blocks is not None:
            self._prior_blocks.append(self._current_blocks)

        # Initialize the new set of blocks
        self._current_blocks = []
        # The identifier for the blockset
        blockset_id = len(self._prior_blocks)
        # We start with the input shape alone
        input_shape = self.input_shape
        for i, gen in enumerate(self._block_generators):
            # Generate the new block
            block = gen(input_shape)
            label = f"blockset_{blockset_id}_layer_{i}"
            self.add_module(label, block)
            self._current_blocks.append(block)
            # Update the output shape for this layer (accounting for concat operation)
            new_output = np.array(block(torch.rand(256, *input_shape)).data.shape[1:])
            new_output[0] = new_output[0] + self._layer_output_shapes[i][0]
            # The below is required because PyCharm has a stroke and thinks integer indexing is a conspiracy
            # noinspection PyTypeChecker
            self._layer_output_shapes[i] = new_output
            # Set the input shape of the next progression block to this new output shape
            input_shape = new_output
            
        # Then, if an output layer builder is specified, re-generate it, replacing the previous one
        if self._output_builder is not None:
            # Direct 'modules' access is required, as the `add_modules` function throws an error trying to replace
            # existing modules with the same name
            self.output_layer = self._output_builder(input_shape)
            self._modules[ProgressiveNN.OUT_LAYER_NAME] = self.output_layer

        # Finally, re-load all the contents of this model back into the selected device
        self.to(device)

    def prune(self, warn=False):
        # If the branch has been frozen, return early and warn the user
        if self._frozen:
            if warn:
                print(f"WARNING: Branch '{self._get_name()}' is frozen, and was not pruned.")
            return

        self.is_pruning = True

        # Apply the pruning method for the branch, filtering down modules to only those with the 'weight' attribute
        if self.update_trained:
            to_prune = [(x, 'weight') for x in self.modules() if hasattr(x, 'weight')]
        else:
            current_block_modules = [b.modules() for b in self._current_blocks]
            current_block_modules.append(self.output_layer.modules())
            to_prune = [(x, 'weight') for x in chain(*current_block_modules) if hasattr(x, 'weight')]
        self._pruning_method(to_prune)

    def forward(self, data):
        # The list of tensor being modified
        prior_tensor_list = []
        current_tensor_list = []

        # Pass the input data through the first set of layers
        first_blocks = [b[0] for b in self._prior_blocks]
        # Newest block appended last, as it produces the final (newest) tensor
        first_blocks.append(self._current_blocks[0])
        for block in first_blocks:
            x = block(data)
            prior_tensor_list.append(x)
        del first_blocks

        # Then pass the remaining inputs through the concatenation chain
        for block_index in range(1, len(self._current_blocks)):
            block_index = block_index
            blocks = [b[block_index] for b in self._prior_blocks]
            # Newest block appended last to received the fully concatenated set of priors
            blocks.append(self._current_blocks[block_index])
            # Chained concatenate of incrementally increasing size
            for i, b in enumerate(blocks):
                x = torch.cat(prior_tensor_list[:(i+1)], dim=1)
                x = b(x)
                current_tensor_list.append(x)
            # Transfer over the accumulated values
            prior_tensor_list = current_tensor_list
            current_tensor_list = []

        # Finally, concatenate the final list of tensors and run it through the output layer (if any)
        out = torch.cat(prior_tensor_list, dim=1)
        if self.output_layer is not None:
            out = self.output_layer(out)

        return out

    def set_frozen(self, state: bool):
        self._frozen = state

    # noinspection PyTypeChecker
    @staticmethod
    def default_l1_pruning(parameters):
        return prune.global_unstructured(
            parameters,
            pruning_method=L1UnstructuredMin1,
            amount=0.1
        )

    @staticmethod
    def default_output_gen(in_shape):
        return


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
    parser.add_argument('-c', '--cycles',
                        help='The number of progressive cycles which should be trained for the model',
                        type=int)
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

    cycles = args.cycles
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

    for cycle in range(cycles):
        print("==========================================================================")
        print(f"Beginning Cycle {cycle}")
        print("==========================================================================")

        base_data = {
            "model": ["Progressive"],
            "cycle": [cycle]
        }

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        testing_dataloader = DataLoader(testing_data, batch_size=batch_size, shuffle=True)

        # Initialize the loss and optimization functions
        loss_fn = nn.CrossEntropyLoss()
        train_optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.99, 0.999), weight_decay=0.001)

        print(f"Beginning Training")

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

        # Restore the final state dict
        # noinspection PyUnboundLocalVariable
        if len(prior_state) != 0 and delta_acc < 0:
            # noinspection PyTypeChecker
            model.load_state_dict(prior_state)

        # Progress the network
        model.progress(device=device)
