import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateNOOP(NamedTuple):
    # fixed input
    g: torch.Tensor

    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    # lengths: torch.Tensor
    # cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        return self.visited_

    @staticmethod
    def initialize(g, visited_dtype=torch.uint8):
        batch_size, n_g, _ = g.size()  # 批次大小,节点数,_
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=g.device)
        return StateNOOP(
            g=g,
            ids=torch.arange(batch_size, dtype=torch.int64, device=g.device)[:, None],
            first_a=prev_a,
            prev_a=prev_a,
            visited_=(
                torch.zeros(
                    batch_size, 1, n_g,
                    dtype=torch.uint8, device=g.device
                )
            ),
            i=torch.zeros(1, dtype=torch.int64, device=g.device)
        )

    def update(self, selected):
        prev_a = selected[:, None]
        first_a = prev_a if self.i.item() == 0 else self.first_a
        visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.g.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited > 0  # Hacky way to return bool or uint8 depending on pytorch version
