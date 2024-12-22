import os
import pathlib

import pathvalidate as pv
import plotly.express as px
import plotly.io as pio
from beartype.typing import Any, Dict, List, Union
from plotly.graph_objs import Figure

from DashAI.back.core.schema_fields import (
    int_field,
    none_type,
    schema_field,
    string_field,
    union_type,
)
from DashAI.back.dataloaders.classes.dashai_dataset import (  # ClassLabel, Value,
    DashAIDataset,
    DatasetDict,
)
from DashAI.back.dependencies.database.models import Exploration, Explorer
from DashAI.back.exploration.base_explorer import BaseExplorer, BaseExplorerSchema


class ParallelCategoriesSchema(BaseExplorerSchema):
    color_column: schema_field(
        none_type(union_type(string_field(), int_field(ge=0))),
        None,
        ("The column to use for coloring the data points. "),
    )  # type: ignore


class ParallelCategoriesExplorer(BaseExplorer):
    """
    Parallel Categories Explorer is a class that generates a parallel categories plot
    for a given dataset.
    """

    DISPLAY_NAME = "Parallel Categories Plot"
    DESCRIPTION = (
        "A parallel categories plot is a common way to visualize "
        "high-dimensional data. "
        "Each vertical line represents one data point, and the lines are connected "
        "by a series of horizontal lines. "
    )

    SCHEMA = ParallelCategoriesSchema
    metadata: Dict[str, Any] = {
        "allowed_dtypes": ["string"],
        "restricted_dtypes": [],
        "input_cardinality": {"min": 2},
    }

    def __init__(self, **kwargs) -> None:
        self.color_column: Union[str, int, None] = kwargs.get("color_column")
        super().__init__(**kwargs)

    def prepare_dataset(
        self, dataset_dict: DatasetDict, columns: List[Dict[str, Any]]
    ) -> DashAIDataset:
        split = list(dataset_dict.keys())[0]
        explorer_columns = [col["columnName"] for col in columns]
        dataset_columns = dataset_dict[split].column_names

        if self.color_column is not None:
            if isinstance(self.color_column, int):
                idx = self.color_column
                col = dataset_columns[idx]
                if col not in explorer_columns:
                    columns.append({"id": idx, "columnName": col})
            else:
                col = self.color_column
                if col not in explorer_columns:
                    columns.append({"columnName": col})
            self.color_column = col

        return super().prepare_dataset(dataset_dict, columns)

    def launch_exploration(self, dataset: DashAIDataset, explorer_info: Explorer):
        _df = dataset.to_pandas()
        columns = [col["columnName"] for col in explorer_info.columns]

        fig = px.parallel_categories(
            _df,
            dimensions=columns,
            color=self.color_column,
            title=(f"Parallel Categories Plot of {len(columns)} columns"),
        )

        if explorer_info.name is not None and explorer_info.name != "":
            fig.update_layout(title=f"{explorer_info.name}")

        return fig

    def save_exploration(
        self,
        __exploration_info__: Exploration,
        explorer_info: Explorer,
        save_path: pathlib.Path,
        result: Figure,
    ) -> str:
        if explorer_info.name is None or explorer_info.name == "":
            filename = f"{explorer_info.id}.json"
        else:
            filename = (
                f"{explorer_info.id}_"
                f"{pv.sanitize_filename(explorer_info.name)}.json"
            )
        path = pathlib.Path(os.path.join(save_path, filename))

        result.write_json(path.as_posix())
        return path.as_posix()

    def get_results(
        self, exploration_path: str, options: Dict[str, Any]
    ) -> Dict[str, Any]:
        resultType = "plotly_json"
        config = {}

        result = pio.read_json(exploration_path)
        result = result.to_json()

        return {"data": result, "type": resultType, "config": config}
