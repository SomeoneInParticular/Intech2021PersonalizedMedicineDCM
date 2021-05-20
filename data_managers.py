import numpy as np
import torchvision.transforms as tt

from cifar100coarse.cifar100coarse import CIFAR100Coarse


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


class PairedMetricCIFAR100Coarse(CIFAR100Coarse):
    """
    A dataset which augments its output with a second set of generated metrics, for use in our PairedDenseCNN
    and (TODO) BranchedProgressiveNN instances
    """

    def __init__(self, seed=None, **kwargs):
        super(PairedMetricCIFAR100Coarse, self).__init__(**kwargs)

        # Cache the prior random state for reproducibility sake, if a specific state was requested
        state = None
        if not seed:
            state = np.random.get_state()
            np.random.seed(seed)

        # The new data consists of three metrics
        self.data2 = np.random.rand(len(self.data), 3)
        self.data2 = self.data2.astype('float32')
        # First metric is left alone

        # Second metric is first scaled down by 0.5
        self.data2[:, 1] = self.data2[:, 1] * 0.5
        # Then offset by 0.25 (making the datas range 0.25->0.75)
        self.data2[:, 1] = self.data2[:, 1] + 0.25
        # Finally, if the category is 2, multiply the metric by 1.33
        self.data2[self.targets == 2, 1] = self.data2[self.targets == 2, 1] * 1.33

        # Third metric initially starts as a metric between 0 and 0.6
        self.data2[:, 2] = self.data2[:, 2] * 0.6
        # It then has the target value (between 0 and 19) multiplied by 0.02 and added to it
        # (Results in a range between 0 and 0.98 for the metric)
        self.data2[:, 2] = self.data2[:, 2] + self.targets * 0.02

        # Restore the random state, if it was explicitly set earlier
        if state:
            np.random.set_state(state)

    def __getitem__(self, item):
        # Get the first entry
        X1, y = super(PairedMetricCIFAR100Coarse, self).__getitem__(item)
        # Fetch the second entry, generated when the dataset was loaded
        X2 = self.data2[item]
        return (X1, X2), y


def build_dataset(train):
    return PairedMetricCIFAR100Coarse(
        root="data",
        train=train,
        download=True,
        transform=train_transform
    )
