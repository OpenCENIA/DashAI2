"""DashAI Dataset implementation."""

import json
import os
import pathlib
from typing import Dict, List, Literal, Tuple, Union

import numpy as np
import polars as pl
import pyarrow as pa
from beartype import beartype
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Value,
    concatenate_datasets,
    load_from_disk,
)
from datasets.table import Table
from polars import DataFrame
from sklearn.model_selection import train_test_split


class DashAIDataset(DataFrame):
    """DashAI dataset wrapper for Huggingface datasets with extra metadata."""

    @beartype
    def __init__(self, df: DataFrame):
        self.df = df
        self.metadata = {}

    @classmethod
    def load(cls, path: Union[str, pathlib.Path]) -> "DashAIDataset":
        """Load dataset from Parquet file"""
        df = pl.read_parquet(path)
        return cls(df)

    def save_to_disk(self, path: Union[str, pathlib.Path]) -> None:
        """Save dataset to Parquet file"""
        self.df.write_parquet(path)

    def change_columns_type(self, column_types: Dict[str, str]) -> "DashAIDataset":
        """Change column types"""
        if not isinstance(column_types, dict):
            raise TypeError(f"types should be a dict, got {type(column_types)}")

        for column in column_types:
            if column in self.column_names:
                pass
            else:
                raise ValueError(
                    f"Error while changing column types: column '{column}' does not "
                    "exist in dataset."
                )

        new_df = self.df.clone()
        for col, dtype in column_types.items():
            if dtype == "Categorical":
                new_df = new_df.with_columns(pl.col(col).cast(pl.Categorical))
            elif dtype == "Numerical":
                new_df = new_df.with_columns(pl.col(col).cast(pl.Float32))
        return DashAIDataset(new_df)

    def remove_columns(self, columns: Union[str, List[str]]) -> "DashAIDataset":
        """Remove specified columns"""
        return DashAIDataset(self.df.drop(columns))

    def sample(
        self,
        n: int = 1,
        method: Literal["head", "tail", "random"] = "head",
        seed: Union[int, None] = None,
    ) -> Dict:
        """Return sample rows"""
        if method == "random":
            sample = self.df.sample(n, seed=seed).to_dict(as_series=False)
        elif method == "head":
            sample = self.df.head(n).to_dict(as_series=False)
        elif method == "tail":
            sample = self.df.tail(n).to_dict(as_series=False)
        return sample

    def __getattr__(self, name):
        """Delegate attribute access to the underlying DataFrame"""
        return getattr(self.df, name)


def load_dataset(path: Union[str, pathlib.Path]) -> DashAIDataset:
    """Load dataset"""
    return DashAIDataset.load(path)


def save_dataset(dataset: DashAIDataset, path: Union[str, pathlib.Path]) -> None:
    """Save dataset"""
    dataset.save_to_disk(path)


@beartype
def split_indexes(
    total_rows: int,
    train_size: float,
    test_size: float,
    val_size: float,
    seed: Union[int, None] = None,
    shuffle: bool = True,
) -> Tuple[List, List, List]:
    """Generate lists with train, test and validation indexes.

    The algorithm for splitting the dataset is as follows:

    1. The dataset is divided into a training and a test-validation split
        (sum of test_size and val_size).
    2. The test and validation set is generated from the test-validation set,
        where the size of the test-validation set is now considered to be 100%.
        Therefore, the sizes of the test and validation sets will now be
        calculated as 100%, i.e. as val_size/(test_size+val_size) and
        test_size/(test_size+val_size) respectively.

    Example:

    If we split a dataset into 0.8 training, a 0.1 test, and a 0.1 validation,
    in the first process we split the training data with 80% of the data, and
    the test-validation data with the remaining 20%; and then in the second
    process we split this 20% into 50% test and 50% validation.

    Parameters
    ----------
    total_rows : int
        Size of the Dataset.
    train_size : float
        Proportion of the dataset for train split (in 0-1).
    test_size : float
        Proportion of the dataset for test split (in 0-1).
    val_size : float
        Proportion of the dataset for validation split (in 0-1).
    seed : Union[int, None], optional
        Set seed to control to enable replicability, by default None
    shuffle : bool, optional
        If True, the data will be shuffled when splitting the dataset,
        by default True.

    Returns
    -------
    Tuple[List, List, List]
        Train, Test and Validation indexes.
    """

    # Generate shuffled indexes
    np.random.seed(seed)
    indexes = np.arange(total_rows)

    test_val = test_size + val_size
    val_proportion = test_size / test_val
    train_indexes, test_val_indexes = train_test_split(
        indexes,
        train_size=train_size,
        random_state=seed,
        shuffle=shuffle,
    )
    test_indexes, val_indexes = train_test_split(
        test_val_indexes,
        train_size=val_proportion,
        random_state=seed,
        shuffle=shuffle,
    )
    return list(train_indexes), list(test_indexes), list(val_indexes)


def split_dataset(
    dataset: DashAIDataset,
    train_size: float,
    test_size: float,
    val_size: float,
    seed: Union[int, None] = None,
    shuffle: bool = True,
) -> Dict[str, DashAIDataset]:
    """Split dataset into train/test/validation"""
    df = dataset.df
    indexes = df.to_series().to_list()

    # Split indexes
    train_idx, temp_idx = train_test_split(
        indexes, train_size=train_size, random_state=seed, shuffle=shuffle
    )

    test_val_size = test_size + val_size
    test_idx, val_idx = train_test_split(
        temp_idx, test_size=val_size / test_val_size, random_state=seed, shuffle=shuffle
    )

    return {
        "train": DashAIDataset(df[train_idx]),
        "test": DashAIDataset(df[test_idx]),
        "validation": DashAIDataset(df[val_idx]),
    }


def to_dashai_dataset(dataset: DatasetDict) -> DatasetDict:
    """
    Convert all datasets within the DatasetDict to DashAIDataset.

    Returns
    -------
    DatasetDict:
        Datasetdict with datasets converted to DashAIDataset.
    """
    print(dataset["train"].features)
    dataset = DashAIDataset(dataset["train"].to_polars())
    print(dataset.schema)
    return dataset


@beartype
def validate_inputs_outputs(
    dataset: DashAIDataset,
    inputs: List[str],
    outputs: List[str],
) -> None:
    """Validate the columns to be chosen as input and output.
    The algorithm considers those that already exist in the dataset.

    Parameters
    ----------
    names : List[str]
        Dataset column names.
    inputs : List[str]
        List of input column names.
    outputs : List[str]
        List of output column names.
    """
    dataset_features = list(dataset.df.columns)
    if len(inputs) == 0 or len(outputs) == 0:
        raise ValueError(
            "Inputs and outputs columns lists to validate must not be empty"
        )
    if len(inputs) + len(outputs) > len(dataset_features):
        raise ValueError(
            "Inputs and outputs cannot have more elements than names. "
            f"Number of inputs: {len(inputs)}, "
            f"number of outputs: {len(outputs)}, "
            f"number of names: {len(dataset_features)}. "
        )
        # Validate that inputs and outputs only contain elements that exist in names
    if not set(dataset_features).issuperset(set(inputs + outputs)):
        raise ValueError(
            f"Inputs and outputs can only contain elements that exist in names. "
            f"Extra elements: "
            f"{', '.join(set(inputs + outputs).difference(set(dataset_features)))}"
        )


@beartype
def get_column_names_from_indexes(
    dataset: DashAIDataset, indexes: List[int]
) -> List[str]:
    """Obtain the column labels that correspond to the provided indexes.

    Note: indexing starts from 1.

    Parameters
    ----------
    datasetdict : DatasetDict
        Path where the dataset is stored.
    indices : List[int]
        List with the indices of the columns.

    Returns
    -------
    List[str]
        List with the labels of the columns
    """

    dataset_features = list(dataset.df.columns)
    col_names = []
    for index in indexes:
        if index > len(dataset_features):
            raise ValueError(
                f"The list of indices can only contain elements within"
                f" the amount of columns. "
                f"Index {index} is greater than the total of columns."
            )
        col_names.append(dataset_features[index - 1])
    return col_names


@beartype
def select_columns(
    dataset: DashAIDataset,
    input_columns: List[str],
    output_columns: List[str],
) -> Tuple[DashAIDataset, DashAIDataset]:
    """Divide the dataset into a dataset with only the input columns in it
    and other dataset only with the output columns

    Parameters
    ----------
    dataset : DashAIDataset
        Dataset to divide
    input_columns : List[str]
        List with the input columns labels
    output_columns : List[str]
        List with the output columns labels

    Returns
    -------
    Tuple[DashAIDataset, DashAIDataset]
        Tuple with the separated DashAIDataset x and y
    """
    input_columns_dataset = dataset.df.select(input_columns)
    output_columns_dataset = dataset.df.select(output_columns)
    return (input_columns_dataset, output_columns_dataset)


@beartype
def get_columns_spec(dataset_path: str) -> Dict[str, str]:
    """Return the column with their respective types

    Parameters
    ----------
    dataset_path : str
        Path where the dataset is stored.

    Returns
    -------
    Dict
        Dict with the columns name and types
    """
    dataset = load_dataset(dataset_path)
    column_types = dataset.df.schema.to_python()
    column_types = {col: str(dtype) for col, dtype in dataset.df.schema.items()}
    return column_types


@beartype
def update_columns_spec(dataset_path: str, columns: Dict) -> DatasetDict:
    """Update the column specification of some dataset on secondary memory.

    Parameters
    ----------
    dataset_path : str
        Path where the dataset is stored.
    columns : Dict
        Dict with columns and types to change

    Returns
    -------
    Dict
        Dict with the columns and types
    """
    if not isinstance(columns, dict):
        raise TypeError(f"types should be a dict, got {type(columns)}")

    # load the dataset from where its stored
    dataframe = load_dataset(dataset_path=dataset_path)
    # copy the features with the columns ans types
    try:
        for column in columns:
            dtype = columns[column]
            dataframe = dataframe.with_column(pl.col(column).cast(dtype))

    except ValueError as e:
        raise ValueError("Error while trying to cast the columns") from e

    return dataframe


def get_dataset_info(dataset_path: str) -> Dict[str, int]:
    """Return the info of the dataset with the number of rows and columns.

    Parameters
    ----------
    dataset_path : str
        Path where the dataset is stored.

    Returns
    -------
    Dict[str, int]
        Dictionary with the information of the dataset
    """
    dataset = load_dataset(dataset_path).df
    total_rows = dataset.height
    total_columns = dataset.width

    dataset_info = {
        "total_rows": total_rows,
        "total_columns": total_columns,
    }
    return dataset_info


@beartype
def update_dataset_splits(
    datasetdict: DatasetDict, new_splits: object, is_random: bool
) -> DatasetDict:
    """Splits an already separated dataset by concatenating it and applying
    new splits. The splits could be random by giving numbers between 0 and 1
    in new_splits parameters and setting the is_random parameter to True, or
    the could be manually selected by giving lists of indices to new_splits
    parameter and setting the is_random parameter to False.

    Args:
        datasetdict (DatasetDict): Dataset to update splits
        new_splits (object): Object with the new train, test and validation config
        is_random (bool): If the new splits are random by percentage

    Returns:
        DatasetDict: New DatasetDict with the new splits configuration
    """
    concatenated_dataset = concatenate_datasets(
        [datasetdict["train"], datasetdict["test"], datasetdict["validation"]]
    )
    n = len(concatenated_dataset)
    if is_random:
        check_split_values(
            new_splits["train"], new_splits["test"], new_splits["validation"]
        )
        train_indexes, test_indexes, val_indexes = split_indexes(
            n, new_splits["train"], new_splits["test"], new_splits["validation"]
        )
    else:
        train_indexes = new_splits["train"]
        test_indexes = new_splits["test"]
        val_indexes = new_splits["validation"]
    return split_dataset(
        dataset=concatenated_dataset,
        train_indexes=train_indexes,
        test_indexes=test_indexes,
        val_indexes=val_indexes,
    )
