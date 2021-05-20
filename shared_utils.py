from itertools import chain
from pathlib import Path

import torch
import torchvision.transforms as tt
import pandas as pd
from torch.nn.utils import prune

stats = ((0.5074, 0.4867, 0.4411), (0.2011, 0.1987, 0.2025))
train_transform = tt.Compose([
    tt.RandomHorizontalFlip(),
    tt.RandomCrop(32, padding=4, padding_mode='reflect'),
    tt.ToTensor(),
    tt.Normalize(*stats)
])
test_transform = tt.Compose([
    tt.ToTensor(),
    tt.Normalize(*stats)
])


def run_train_cycle(model, dataloader, loss_fn, optim, device):
    # Set the model to train mode
    model.train()

    train_loss, train_acc = 0, 0

    for i, (X, y) in enumerate(dataloader):
        # Load the data into the device for processing
        X, y = X.to(device), y.to(device)

        # Prediction loss calc
        pred = model(X)
        loss = loss_fn(pred, y)

        # Evaluate the loss and accuracy for this batch
        train_loss += loss.item()
        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation to update the model
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Report loss every 50 batches
        if i % 50 == 0:
            print(f'Loss: {loss.item()} for batch {i}')

    size = len(dataloader.dataset)
    train_loss /= size
    train_acc /= size

    return {
        "loss": [train_loss],
        "accuracy": [train_acc]
    }


def run_test_cycle(model, dataloader, loss_fn, device):
    # Set this to evaluate mode
    model.eval()
    # Accumulate loss and accuracy across the dataset
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            # Load the data onto the device and have the model predict it
            X, y = X.to(device), y.to(device)
            pred = model(X)

            # Evaluate the total loss and accuracy for the batch
            test_loss += loss_fn(pred, y).item()
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Evaluate the average loss and accuracy across the dataset
    size = len(dataloader.dataset)
    test_loss /= size
    test_acc /= size
    # Report it to the the console
    print(f"Test Error: \n Accuracy: {(100 * test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # Return a dictionary containing the results (for use in plotting etc.), list wrapped for dataframe usage
    return {
        "loss": [test_loss],
        "accuracy": [test_acc]
    }


def default_lateral(x):
    return torch.cat(x, dim=1)


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


def initialize_output_files(out_path: Path):
    header = ["Cycle", "Epoch", "Accuracy", "Loss"]
    df = pd.DataFrame(columns=header)
    train_file = Path(out_path, "training.tsv")
    df.to_csv(train_file, mode='w', sep='\t', header=True)
    valid_file = Path(out_path, "validation.tsv")
    df.to_csv(valid_file, mode='w', sep='\t', header=True)


def write_new_result(cycle, epoch, accuracy, loss, out_path, is_train: True):
    df = pd.DataFrame({
        "Cycle": cycle,
        "Epoch": epoch,
        "Accuracy": accuracy,
        "Loss": loss
    })
    target_file = "training.tsv" if is_train else "validation.tsv"
    target_path = Path(out_path, target_file)
    df.to_csv(target_path, mode='a', header=False, sep='\t')


def default_l1_prune(model):
    params_to_prune = [(b, 'weight') for b in model.columns[-1].modules() if hasattr(b, 'weight')]
    prune.global_unstructured(
        params_to_prune,
        pruning_method=L1UnstructuredMin1,
        amount=0.1
    )

def default_full_l1_prune(model):
    params_to_prune = [(b, 'weight') for b in chain(*[c.modules() for c in model.columns]) if hasattr(b, 'weight')]
    prune.global_unstructured(
        params_to_prune,
        pruning_method=L1UnstructuredMin1,
        amount=0.1
    )
