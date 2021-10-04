from typing import Iterator
from typing import Optional

from torch.utils.data import DataLoader


class BatchNormTuneDataLoaderWrapper:
    """
    Wrapper over torch.utils.data.DataLoader to enhance batch norm tuning procedure speed.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Batch norm tuning dataloader.
    """

    def __init__(self, dataloader: DataLoader):
        self.dataloader: DataLoader = dataloader
        self.length: int = len(dataloader)
        self.current_iter: Optional[Iterator] = None

    def __iter__(self) -> Iterator:

        if self.current_iter is None:
            self.current_iter = iter(self.dataloader)

        while True:
            try:
                yield next(self.current_iter)
            except StopIteration:
                self.current_iter = iter(self.dataloader)
                yield next(self.current_iter)

    def __len__(self) -> int:
        """
        Returns the length of user dataloader.

        Returns
        -------
        int
            User dataloader's length.

        """
        return self.length

