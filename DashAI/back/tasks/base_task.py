from abc import abstractmethod
from typing import Any, Dict, Final, List, Union

from datasets import DatasetDict

from DashAI.back.dataloaders.classes.dashai_dataset import DashAIDataset


class BaseTask:
    """Base class for DashAI compatible tasks."""

    TYPE: Final[str] = "Task"

    @property
    @abstractmethod
    def schema(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Get metadata values for the current task

        Returns:
            Dict[str, Any]: Dictionary with the metadata containing the input and output
             types/cardinality.
        """
        metadata = cls.metadata

        # Extract class names
        inputs_types = [input_type.__name__ for input_type in metadata["inputs_types"]]
        outputs_types = [
            output_type.__name__ for output_type in metadata["outputs_types"]
        ]

        parsed_metadata: dict = {
            "inputs_types": inputs_types,
            "outputs_types": outputs_types,
            "inputs_cardinality": metadata["inputs_cardinality"],
            "outputs_cardinality": metadata["outputs_cardinality"],
        }
        return parsed_metadata

    def validate_dataset_for_task(
        self,
        dataset: DashAIDataset,
        dataset_name: str,
        input_columns: List[str],
        output_columns: List[str],
    ) -> None:
        """Validate a dataset for the current task.

        Parameters
        ----------
        dataset : DatasetDict
            Dataset to be validated
        dataset_name : str
            Dataset name
        """
        metadata = self.metadata
        allowed_input_types = tuple(metadata["inputs_types"])
        allowed_output_types = tuple(metadata["outputs_types"])
        inputs_cardinality = metadata["inputs_cardinality"]
        outputs_cardinality = metadata["outputs_cardinality"]

        # Check input types
        for input_col in input_columns:
            input_col_type = dataset.features[input_col]
            if not isinstance(input_col_type, allowed_input_types):
                raise TypeError(
                    f"{input_col_type} is not an allowed type for input columns."
                )

        # Check output types
        for output_col in output_columns:
            output_col_type = dataset.features[output_col]
            if not isinstance(output_col_type, allowed_output_types):
                raise TypeError(
                    f"{output_col_type} is not an allowed type for output columns."
                )

        # Check input cardinality
        if inputs_cardinality != "n" and len(input_columns) != inputs_cardinality:
            raise ValueError(
                f"Input cardinality ({len(input_columns)}) does not"
                f" match task cardinality ({inputs_cardinality})"
            )

        # Check output cardinality
        if outputs_cardinality != "n" and len(output_columns) != outputs_cardinality:
            raise ValueError(
                f"Output cardinality ({len(output_columns)})"
                f" does not "
                f"match task cardinality ({outputs_cardinality})"
            )

    @abstractmethod
    def prepare_for_task(
        self, dataset: Union[DatasetDict, DashAIDataset], outputs_columns: List[str]
    ) -> Union[DatasetDict, DashAIDataset]:
        """Change column types to suit the task requirements.

        Parameters
        ----------
        dataset : Union[DatasetDict, DashAIDataset]
            Dataset to be changed

        Returns
        -------
        Union[DatasetDict, DashAIDataset]
            Dataset with the new types
        """
        raise NotImplementedError
