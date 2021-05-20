from typing import Callable, List

from torch import nn


# Code based heavily on:
# https://towardsdatascience.com/progressive-neural-networks-explained-implemented-6f07366d714d


class ProgBlock(nn.Module):
    """
    A progressive learning block, which manages the concatenation of
    inputs, alongside the normal functions of a module
    """

    def __init__(self, module, lateral):
        super().__init__()
        self.block = module
        self.lateral = lateral

    def run_block(self, x):
        return self.block(x)

    def run_lateral(self, *data):
        return self.lateral(data)

    def forward(self, x):
        return self.block(x)


class ProgColumn(nn.Module):
    """
    A column within a progressive learning network, generally one per
    task. Tracks the calculated tensors during its forward operation,
    allowing them to be fed into other columns within the full network
    """
    def __init__(self, block_list: List[ProgBlock],
                 parent_cols: List['ProgColumn'] = None):
        super().__init__()
        self.blocklist = nn.ModuleList(block_list)
        if parent_cols:
            self.parent_cols = parent_cols
        else:
            self.parent_cols = []
        self.no_rows = len(block_list)
        self.last_output_list = []
        self.isFrozen = False

    def freeze(self, unfreeze=False):
        if not unfreeze:
            self.isFrozen = True
            [setattr(p, 'requires_grad', False) for p in self.parameters()]
        else:
            self.isFrozen = False
            [setattr(p, 'requires_grad', True) for p in self.parameters()]

    def forward(self, x):
        new_outputs = []

        for row, block in enumerate(self.blocklist):
            # If this is *not* the first block and has parent cols,
            # run lateral on the parent cols outputs first
            if row != 0 and len(self.parent_cols) > 0:
                tensor_list = [c.last_output_list[row-1] for c in self.parent_cols]
                x = block.run_lateral(x, *tensor_list)
            # Run the block's actual contents
            x = block(x)
            # Save the output for other columns to hook into later
            new_outputs.append(x)
        self.last_output_list = new_outputs
        return new_outputs[-1]


class ProgNet(nn.Module):
    """
    The actual progressive model, using the modifications dictated in
    Hayek et. al 2020
    """
    def __init__(self, column_generator: Callable[[List[nn.Module]], ProgColumn] = None,
                 device="cpu"):
        super().__init__()
        self.columns = nn.ModuleList().to(device)
        self.no_rows = None
        self.no_cols = 0
        self.col_gen = column_generator
        self.col_shape = None
        self.device = device

    def add_column(self, col: ProgColumn = None):
        if not col:
            parents = [ref for ref in self.columns]
            col = self.col_gen(parents)
        col.to(device=self.device)
        self.columns.append(col)
        self.no_rows = col.no_rows
        self.no_cols += 1

    def freeze_column(self, index: int):
        col = self.columns[index]
        col.freeze()

    def freeze_all_columns(self):
        [c.freeze() for c in self.columns]

    def unfreeze_column(self, index: int):
        col = self.columns[index]
        col.freeze(unfreeze=True)

    def unfreeze_all_columns(self):
        [c.freeze(unfreeze=True) for c in self.columns]

    def get_column(self, index: int):
        return self.columns[index]

    def forward(self, x):
        y = None
        # Iterate through the columns one at a time
        for i, col in enumerate(self.columns):
            y = col(x)
        # The final columns is always the newest, having the output
        return y
