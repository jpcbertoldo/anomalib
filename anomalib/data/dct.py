""" Diffraction Contrast Tomography (DCT)

# flake8: noqa: E501
# pylint: disable=line-too-long
https://www.esrf.fr/home/UsersAndScience/Experiments/StructMaterials/ID11/techniques/diffraction-contrast-tomography.html

TODO: this should evolve to a separate HDF5 reader that should be optionally pre-load or read on the fly (important feature for big high-resolution datasets)
this one would be a composition of three of those (raw, ref, dark)

TODO: dont' forget to get the log(data)

"""
from typing import Optional

from pytorch_lightning.core.datamodule import LightningDataModule
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.datasets.folder import VisionDataset


def get_array_from_hdf5():
    pass


def make_dataset_from_dct_acquisition():
    """Make dataset from DCT acquisition HDF5 file

    The expected HDF5 file structure (at the given path) is:
    data
    |-raw
    |-ref
    |-dark

    obs: make sure to use a fully qualified path if possible
    i.e. the file path and the internal hdf5 path (group)

    this is called url in silx
    ex: DataUrl("silx:///data/image.h5::/data/dataset")

    the vocabulary there is: f"{scheme}://{file_path}::{data_path}"

    https://github.com/silx-kit/silx/blob/77c6893ec3dbbe18b82e8e7666cf27acd2279db1/src/silx/io/url.py#L64

    i could use a simplified version of this without the scheme

    file_path = '/path/to/file.hdf5'
    data_path = '/res256'

    with h5py.File(file_path) as h5:

        # print(h5.keys())
        # <KeysViewHDF5 ['data_256', 'data_512']>

        data = h5[data_path]

        # print(data.keys())
        # <KeysViewHDF5 ['dark', 'raw', 'ref']>

        raw: np.ndarray = data['raw'][()]
        ref: np.ndarray = data['ref'][()]
        dark: np.ndarray = data['dark'][()]
    """
    pass


class DiffractionContrastTomographyDataset(VisionDataset):
    pass

    def __init__():
        pass

    def __len__(self) -> int:
        pass

    def __getitem__(self, index: int):
        pass


@DATAMODULE_REGISTRY
class DiffractionContrastTomographyDataModule(LightningDataModule):
    pass

    def __init__(
        self,
        num_workers: int = 8,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
    ):
        self.num_workers = num_workers
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size

        self.train_data: Optional[Dataset] = None
        self.val_data: Optional[Dataset] = None
        self.test_data: Optional[Dataset] = None
        self.inference_data: Optional[Dataset] = None

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """Get train dataloader."""
        return DataLoader(self.train_data, shuffle=True, batch_size=self.train_batch_size, num_workers=self.num_workers)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """Get test dataloader."""
        return DataLoader(self.test_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        """Get predict dataloader."""
        return DataLoader(
            self.inference_data, shuffle=False, batch_size=self.test_batch_size, num_workers=self.num_workers
        )
