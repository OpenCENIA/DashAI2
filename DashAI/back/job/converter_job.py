import logging
import os
import re
from importlib import import_module
from typing import Dict, List, Union

import pandas as pd
from datasets import Dataset, DatasetDict
from kink import inject
from pydantic import BaseModel as PydanticBaseModel
from sqlalchemy import exc
from sqlalchemy.orm import Session

from DashAI.back.dataloaders.classes.dashai_dataset import (
    DashAIDataset,
    load_dataset,
    save_dataset,
    to_dashai_dataset,
)
from DashAI.back.dependencies.database.models import ConverterList
from DashAI.back.dependencies.database.models import Dataset as DatasetModel
from DashAI.back.dependencies.registry import ComponentRegistry
from DashAI.back.job.base_job import BaseJob, JobError

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class ConverterParams(PydanticBaseModel):
    params: Dict[str, Union[str, int, float, bool]] = None
    scope: Dict[str, List[int]] = None


class ConverterListJob(BaseJob):
    """ConverterListJob class to modify a dataset by applying a sequence of converters."""

    def set_status_as_delivered(self) -> None:
        """Set the status of the list as delivered."""
        converter_list_id: int = self.kwargs["converter_list_id"]
        db: Session = self.kwargs["db"]
        converter_list: ConverterList = db.get(ConverterList, converter_list_id)
        if converter_list is None:
            raise JobError(
                f"Converter list with id {converter_list_id} does not exist in DB."
            )
        try:
            converter_list.set_status_as_delivered()
            db.commit()
        except exc.SQLAlchemyError as e:
            log.exception(e)
            raise JobError(
                "Error while setting the status of the converter list as delivered."
            ) from e

    @inject
    def run(
        self,
        component_registry: ComponentRegistry = lambda di: di["component_registry"],
    ) -> None:
        converter_list_id: int = self.kwargs["converter_list_id"]
        target_column_index: int = self.kwargs["target_column_index"]
        db: Session = self.kwargs["db"]

        # Load the converter list information
        try:
            if converter_list_id is None or target_column_index is None:
                raise JobError(
                    "Converter list ID and target column index are required to run the job."
                )

            converter_list: ConverterList = db.get(ConverterList, converter_list_id)
            if not converter_list:
                raise JobError(
                    f"Converter list with id {converter_list_id} does not exist in DB."
                )
            
            converter_list.set_status_as_started()
            db.commit()
        except exc.SQLAlchemyError as e:
            log.exception(e)
            raise JobError("Error while loading the converter list info.") from e

        # Load dataset information
        try:
            dataset_id = converter_list.dataset_id
            dataset: DatasetModel = db.get(DatasetModel, dataset_id)
            if not dataset:
                raise JobError(f"Dataset with id {dataset_id} does not exist in DB.")
        except exc.SQLAlchemyError as e:
            log.exception(e)
            converter_list.set_status_as_error()
            db.commit()
            raise JobError("Error while loading the dataset info.") from e

        # Load dataset
        try:
            dataset_path = f"{dataset.file_path}/dataset"
            dataset_dict = load_dataset(dataset_path, keep_in_memory=True)
            if int(target_column_index) < 1 or int(target_column_index) > len(
                dataset_dict["train"].features
            ):
                raise JobError(
                    f"Target column index {target_column_index} is out of bounds."
                )
        except Exception as e:
            log.exception(e)
            converter_list.set_status_as_error()
            db.commit()
            raise JobError(
                f"Can not load dataset from path {dataset_path}"
            ) from e

        try:
            # Regex to convert camel case to snake case
            camel_to_snake = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

            # Create a dictionary with the submodule for each converter
            converters_list_dir = os.listdir("DashAI/back/converters")
            existing_submodules = [
                submodule
                for submodule in converters_list_dir
                if os.path.isdir(f"DashAI/back/converters/{submodule}")
            ]
            converter_submodule_inverse_index = {}
            for submodule in existing_submodules:
                existing_converters = os.listdir(f"DashAI/back/converters/{submodule}")
                for file in existing_converters:
                    if file.endswith(".py"):
                        converter_name = file[:-3]
                        converter_submodule_inverse_index[converter_name] = submodule

            converters_to_apply: Dict[str, ConverterParams] = converter_list.converters
            dataset_original_columns = dataset_dict["train"].column_names

            for converter_name in converters_to_apply:
                dataset_train: DashAIDataset = dataset_dict["train"]
                df_train = dataset_train.to_pandas()
                dataset_test: DashAIDataset = dataset_dict["test"]
                df_test = dataset_test.to_pandas()
                dataset_validation: DashAIDataset = dataset_dict["validation"]
                df_validation = dataset_validation.to_pandas()
                df_concatenated = pd.concat([df_train, df_test, df_validation], axis=0)

                # Get converter constructor and parameters
                converter_filename = camel_to_snake.sub(
                    "_", converter_name
                ).lower()  # CamelCase to snake_case
                submodule = converter_submodule_inverse_index[converter_filename]
                module_path = f"DashAI.back.converters.{submodule}.{converter_filename}"

                # Import the converter
                module = import_module(module_path)
                converter_constructor = getattr(module, converter_name)
                converter_parameters = (
                    converters_to_apply[converter_name]["params"]
                    if "params" in converters_to_apply[converter_name].keys()
                    else {}
                )

                # Create and apply converter
                converter = converter_constructor(**converter_parameters)
                converter_scope = (
                    converters_to_apply[converter_name]["scope"]
                    if "scope" in converters_to_apply[converter_name].keys()
                    else {}
                )

                # Process columns scope
                columns_scope = [
                    column - 1 for column in converter_scope["columns"]
                ]
                scope_column_indexes = list(set(columns_scope))
                scope_column_indexes.sort()
                if scope_column_indexes == []:
                    scope_column_indexes = list(range(len(dataset_train.features)))
                scope_column_names = [
                    dataset_original_columns[index] for index in scope_column_indexes
                ]

                # Process rows scope
                rows_scope = [
                    row - 1 for row in converter_scope["rows"]
                ]
                scope_rows_indexes = list(set(rows_scope))
                scope_rows_indexes.sort()
                if scope_rows_indexes == []:
                    scope_rows_indexes = list(range(len(df_concatenated)))

                target_column_index = int(target_column_index) - 1
                target_column_name = dataset_original_columns[target_column_index]

                # Fit converter
                X = df_concatenated[scope_column_names].iloc[scope_rows_indexes]
                if len(X.shape) == 1:
                    X = X.to_frame()
                y = df_concatenated[target_column_name].iloc[scope_rows_indexes]
                converter = converter.fit(X, y)

                # Transform data
                X = df_concatenated[scope_column_names]
                y = df_concatenated[target_column_name]
                if len(X.shape) == 1:
                    X = X.to_frame()
                resulting_dataframe = converter.transform(X, y)

                # Update dataframe
                columns_to_drop = df_concatenated.columns[scope_column_indexes]
                df_concatenated.drop(columns_to_drop, axis=1, inplace=True)
                for i, column in enumerate(resulting_dataframe.columns):
                    df_concatenated.insert(
                        scope_column_indexes[i], column, resulting_dataframe[column]
                    )

                # Update splits
                df_train = df_concatenated.iloc[: len(dataset_train)]
                df_test = df_concatenated.iloc[
                    len(dataset_train) : len(dataset_train) + len(dataset_test)
                ]
                df_validation = df_concatenated.iloc[
                    len(dataset_train) + len(dataset_test) :
                ]

                # Create final dataset
                dataset_dict = DatasetDict(
                    {
                        "train": Dataset.from_pandas(df_train, preserve_index=False),
                        "test": Dataset.from_pandas(df_test, preserve_index=False),
                        "validation": Dataset.from_pandas(
                            df_validation, preserve_index=False
                        ),
                    }
                )
                dataset_dict = to_dashai_dataset(dataset_dict)

            # Save the final dataset
            save_dataset(dataset_dict, f"{dataset_path}")
            converter_list.set_status_as_finished()
            db.commit()
            db.refresh(dataset)

        except Exception as e:
            log.exception(e)
            converter_list.set_status_as_error()
            db.commit()
            raise JobError(
                f"Error while applying converters to dataset with id {dataset_id}. Error: {e}"
            ) from e