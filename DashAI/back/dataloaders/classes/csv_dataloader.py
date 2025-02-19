"""DashAI CSV Dataloader."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Union

from beartype import beartype
from datasets import DatasetDict, load_dataset
from starlette.datastructures import UploadFile

from DashAI.back.core.schema_fields import (
    bool_field,
    enum_field,
    none_type,
    schema_field,
    string_field,
)
from DashAI.back.core.schema_fields.base_schema import BaseSchema
from DashAI.back.dataloaders.classes.dashai_dataset import (
    DashAIDataset,
    to_dashai_dataset,
)
from DashAI.back.dataloaders.classes.dataloader import BaseDataLoader


class CSVDataloaderSchema(BaseSchema):
    name: schema_field(
        none_type(string_field()),
        "",
        (
            "Custom name to register your dataset. If no name is specified, "
            "the name of the uploaded file will be used."
        ),
    )  # type: ignore
    separator: schema_field(
        enum_field([",", ";", "\u0020", "\t"]),
        ",",
        "A separator character delimits the data in a CSV file.",
    )  # type: ignore
    dataset_is_already_split: schema_field(
        bool_field(),
        False,
        (
            "If your dataset is already split into training, validation, and test sets, "
            "upload it as a ZIP file with each split in separate folders. Otherwise, "
            "upload a single dataset file."
        ),
    )  # type: ignore


class CSVDataLoader(BaseDataLoader):
    """Data loader for tabular data in CSV files."""

    COMPATIBLE_COMPONENTS = ["TabularClassificationTask"]
    SCHEMA = CSVDataloaderSchema

    def _check_params(
        self,
        params: Dict[str, Any],
    ) -> None:
        if "separator" not in params:
            raise ValueError(
                "Error trying to load the CSV dataset: "
                "separator parameter was not provided."
            )
        separator = params["separator"]

        if not isinstance(separator, str):
            raise TypeError(
                f"Param separator should be a string, got {type(params['separator'])}"
            )

    @beartype
    def load_data(
        self,
        filepath_or_buffer: Union[UploadFile, str],
        temp_path: str,
        params: Dict[str, Any],
    ) -> DashAIDataset:
        """Load the uploaded CSV files into a DatasetDict.

        Parameters
        ----------
        filepath_or_buffer : Union[UploadFile, str], optional
            An URL where the dataset is located or a FastAPI/Uvicorn uploaded file
            object.
        temp_path : str
            The temporary path where the files will be extracted and then uploaded.
        params : Dict[str, Any]
            Dict with the dataloader parameters. The options are:
            - `separator` (str): The character that delimits the CSV data.

        Returns
        -------
        DatasetDict
            A HuggingFace's Dataset with the loaded data.
        """
        self._check_params(params)
        separator = params["separator"]
        prepared_path = self.prepare_files(filepath_or_buffer, temp_path)
        if prepared_path[1] == "file":
            dataset = load_dataset(
                "csv",
                data_files=prepared_path[0],
                delimiter=separator,
            )
        else:
            dataset = load_dataset(
                "csv",
                data_dir=prepared_path[0],
                delimiter=separator,
            )

        return to_dashai_dataset(dataset)
