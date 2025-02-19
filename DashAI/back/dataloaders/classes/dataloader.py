"""DashAI base class for dataloaders."""

import io
import logging
import os
import zipfile
from abc import abstractmethod
from typing import Any, Dict, Final, Union

from datasets.download.download_manager import DownloadManager as dl_manager
from starlette.datastructures import UploadFile

from DashAI.back.config_object import ConfigObject
from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset

logger = logging.getLogger(__name__)


"""
class DatasetSplitsSchema(BaseSchema):
    train_size: schema_field(
        float_field(ge=0.0, le=1.0),
        0.7,
        (
            "The training set contains the data to be used for training a model. "
            "Must be defined between 0 and 100% of the data."
        ),
    )  # type: ignore
    test_size: schema_field(
        float_field(ge=0.0, le=1.0),
        0.2,
        (
            "The test set contains the data that will be used to evaluate a model. "
            "Must be defined between 0 and 100% of the data."
        ),
    )  # type: ignore
    val_size: schema_field(
        float_field(ge=0.0, le=1.0),
        0.1,
        (
            "The validation set contains the data to be used to validate a model. "
            "Must be defined between 0 and 100% of the data."
        ),
    )  # type: ignore


class DataloaderMoreOptionsSchema(BaseSchema):
    shuffle: schema_field(
        bool_field(),
        True,
        (
            "Determines whether the data will be shuffle when defining the sets or "
            "not. It must be true for shuffle the data, otherwise false."
        ),
    )  # type: ignore
    seed: schema_field(
        int_field(ge=0),
        0,
        (
            "A seed defines a value with which the same mixture of data will always "
            "be obtained. It must be an integer greater than or equal to 0."
        ),
    )  # type: ignore
    stratify: schema_field(
        bool_field(),
        False,
        (
            "Defines whether the data will be proportionally separated according to "
            "the distribution of classes in each set."
        ),
    )  # type: ignore

"""


class BaseDataLoader(ConfigObject):
    """Abstract class with base methods for DashAI dataloaders."""

    TYPE: Final[str] = "DataLoader"

    @abstractmethod
    def load_data(
        self,
        filepath_or_buffer: Union[UploadFile, str],
        temp_path: str,
        params: Dict[str, Any],
    ) -> DashAIDataset:
        """Load data abstract method.

        Parameters
        ----------
        filepath_or_buffer : Union[UploadFile, str], optional
            An URL where the dataset is located or a FastAPI/Uvicorn uploaded file
            object.
        temp_path : str
            The temporary path where the files will be extracted and then uploaded.
        params : Dict[str, Any]
            Dict with the dataloader parameters.

        Returns
        -------
        DashAIDataset
            A DashAI Dataset with the loaded data.
        """
        raise NotImplementedError

    def prepare_files(self, file_path: str, temp_path: str) -> str:
        """Prepare the files to load the data.

        Args:
            file_path (str): Path of the file to be prepared.
            temp_path (str): Temporary path where the files will be extracted.

        Returns
        -------

            path (str): Path of the files prepared.
            type_path (str): Type of the path.

        """
        if file_path.startswith("http"):
            file_path = dl_manager.download_and_extract(file_path, temp_path)
            return (file_path, "dir")

        if isinstance(file_path, UploadFile):
            local_file_path = os.path.join(temp_path, file_path.filename)
            with open(local_file_path, "wb") as f:
                f.write(file_path.file.read())
            file_path = local_file_path

        if file_path.lower().endswith(".zip"):
            extracted_path = self.extract_files(
                file_path=file_path, temp_path=temp_path
            )
            return (extracted_path, "dir")

        else:
            return (file_path, "file")

    def extract_files(self, file_path: str, temp_path: str) -> str:
        """Extract the files to load the data in a DataDict later.

        Args:
            dataset_path (str): Path where dataset will be saved.
            file (UploadFile): File uploaded for the user.

        Returns
        -------
            str: Path of the files extracted.
        """
        files_path = os.path.join(temp_path, "files")
        os.makedirs(files_path, exist_ok=True)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(files_path)
        return files_path
