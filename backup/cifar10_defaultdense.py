import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, random_split
from torchvision.models.densenet import densenet169

import numpy as np
import data_managers
from cifar100coarse.cifar100coarse import CIFAR100Coarse


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
        pred = model(X)

        # Backpropagation to update the model
        optim.zero_grad()
        loss = loss_fn(pred, y)
        losses.append(float(loss.cpu().detach().numpy()))
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
    train_epochs = 50

    device = "cpu"

    model = densenet169()

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

    optim = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001, momentum=0.9, dampening=0)

    loss_fn = nn.CrossEntropyLoss()

    scheduler = MultiStepLR(optim, milestones=[
        train_epochs // 2,
        train_epochs * 3 // 4
    ], gamma=0.1)

    for i in range(len(training_sets)):

        train_dataloader = DataLoader(training_sets[i], batch_size=64, shuffle=True)
        testing_dataloader = DataLoader(testing_sets[i], batch_size=64, shuffle=True)

        def run_epochs(target_model, epoch_no, optim, scheduler, loss):
            # Runs epoch_no epochs on the target model, using the provided optimizer to update the model and
            # the provided loss function to evaluate how "good" it during evaluation
            results = []
            for e in range(epoch_no):
                print(f"Epoch {e + 1}\n----------------------------------------------------")

                # Training
                run_train_cycle(target_model, train_dataloader, loss, optim, device)

                # Testing/Validation
                test_results = run_test_cycle(target_model, testing_dataloader, loss, device)
                results.append(test_results)

                # Scheduler update
                scheduler.step()
            return results


        run_epochs(model, train_epochs, optim, scheduler, loss_fn)
