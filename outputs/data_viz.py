import matplotlib.pyplot as plt

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


def plot_all_data(full_training_iter, full_validation_iter,
                  batched_training_iter, batched_validation_iter,
                  cycle_epoch):
    # Prepare the parameters needed for the plotting
    no_rows = 2
    no_cols = len(full_training_iter)
    line_width = 0.6
    font_size = 8

    # Initialize the plotting space
    fig, axs = plt.subplots(no_rows, no_cols)

    # Begin the plotting
    for i, k in enumerate(sorted(full_training_iter.keys())):
        # Plot our full data setup (top)
        ax_full = axs[0][i]
        train_df = pd.read_csv(full_training_iter[k], sep='\t')
        valid_df = pd.read_csv(full_validation_iter[k], sep='\t')
        ax_full.plot(train_df.index, train_df['Accuracy'], color='C0',
                     linewidth=line_width)
        ax_full.plot(valid_df.index, valid_df['Accuracy'], color='C1',
                     linewidth=line_width)

        # Plot vertical lines at the maximum value of training and testing
        train_max = max(train_df['Accuracy'])
        ax_full.hlines(train_max, 0, 1,
                       transform=ax_full.get_yaxis_transform(),
                       colors='C0', linewidths=line_width, linestyles='dotted')
        valid_max = max(valid_df['Accuracy'])
        ax_full.hlines(valid_max, 0, 1,
                       transform=ax_full.get_yaxis_transform(),
                       colors='C1', linewidths=line_width, linestyles='dashed')

        # Plot horizontal lines at the end of each cycle
        ax_full.vlines(valid_df[valid_df['Epoch'] == cycle_epoch].index, 0, 1,
                       transform=ax_full.get_xaxis_transform(),
                       colors='black', linewidths=line_width,
                       linestyles=(0, (1, 5)))

        # Plot our batched setup (bottom)
        ax_batch = axs[1][i]

        train_df = pd.read_csv(batched_training_iter[k], sep='\t')
        valid_df = pd.read_csv(batched_validation_iter[k], sep='\t')
        ax_batch.plot(train_df.index, train_df['Accuracy'], color='C0',
                      linewidth=line_width)
        ax_batch.plot(valid_df.index, valid_df['Accuracy'], color='C1',
                      linewidth=line_width)

        # Plot vertical lines at the maximum value of training and testing
        train_max = max(train_df['Accuracy'])
        ax_batch.hlines(train_max, 0, 1, transform=ax_batch.get_yaxis_transform(),
                        colors='C0', linewidths=line_width, linestyles='dotted')
        valid_max = max(valid_df['Accuracy'])
        ax_batch.hlines(valid_max, 0, 1, transform=ax_batch.get_yaxis_transform(),
                        colors='C1', linewidths=line_width, linestyles='dashed')

        # Plot horizontal lines at the end of each cycle
        ax_batch.vlines(valid_df[valid_df['Epoch'] == cycle_epoch].index, 0, 1,
                        transform=ax_batch.get_xaxis_transform(),
                        colors='black', linewidths=line_width,
                        linestyles=(0, (1, 5)))

        # Apply final formatting to the plots
        if "free" in k:
            title = "Progressive, Free"
        elif "prog" in k:
            title = "Progressive, Frozen"
        else:
            title = "Static"
        # Only title the top row
        ax_full.set_title(title)

        # The y range should all be shared between 0% and 100%
        ax_full.set_ylim(bottom=0., top=1.)
        ax_batch.set_ylim(bottom=0., top=1.)

        # Only add y-axis labels to the leftmost row
        if i == 0:
            ax_full.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax_full.tick_params(axis='y', labelsize=font_size)
            ax_full.set_ylabel("Full Data", fontsize=font_size)

            ax_batch.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax_batch.tick_params(axis='y', labelsize=font_size)
            ax_batch.set_ylabel("Batched Data", fontsize=font_size)
        else:
            ax_full.set_yticklabels([])
            ax_batch.set_yticklabels([])

        # Only add the x-axis label to the center bottom row (batch data)
        if i == no_cols//2:
            ax_batch.set_xlabel("Training Epoch", fontsize=font_size)

        # Rescale the tick size of the epoch counter to match the y axis
        ax_full.tick_params(axis='x', labelsize=font_size)
        ax_batch.tick_params(axis='x', labelsize=font_size)

    return fig


if __name__ == '__main__':
    # Prepare our full_set data paths
    full_set_trainings = {str(p.parent).split('/')[-1]: p
                          for p in Path('full_set').glob('*/training.tsv')}
    full_set_validations = {str(p.parent).split('/')[-1]: p
                            for p in Path('full_set').glob('*/validation.tsv')}

    # # Prepare our batched data paths
    batched_trainings = {str(p.parent).split('/')[-1]: p
                         for p in Path('batched').glob('*/training.tsv')}
    batched_validations = {str(p.parent).split('/')[-1]: p
                           for p in Path('batched').glob('*/validation.tsv')}

    # Plot in two batches; one for the ConvNet, one for the DenseNet
    convnet_full_train = {k: v for k, v in full_set_trainings.items()
                          if "conv2d" in k}
    convnet_batch_train = {k: v for k, v in batched_trainings.items()
                           if "conv2d" in k}

    convnet_full_test = {k: v for k, v in full_set_validations.items()
                         if "conv2d" in k}
    convnet_batch_test = {k: v for k, v in batched_validations.items()
                          if "conv2d" in k}

    convnet_fig = plot_all_data(convnet_full_train, convnet_full_test,
                                convnet_batch_train, convnet_batch_test,
                                89)
    convnet_fig.suptitle("ConvNet Training Processes")
    plt.tight_layout()
    plt.savefig("./convent_results.pdf", pad_inches='tight')

    dense_full_train = {k: v for k, v in full_set_trainings.items()
                        if "dense" in k}
    dense_batch_train = {k: v for k, v in batched_trainings.items()
                         if "dense" in k}

    dense_full_test = {k: v for k, v in full_set_validations.items()
                         if "dense" in k}
    dense_batch_test = {k: v for k, v in batched_validations.items()
                          if "dense" in k}

    dense_fig = plot_all_data(dense_full_train, dense_full_test,
                              dense_batch_train, dense_batch_test,
                              299)
    dense_fig.suptitle("DenseNet Training Processes")
    plt.tight_layout()
    plt.savefig("./dense_results.pdf", pad_inches='tight')
